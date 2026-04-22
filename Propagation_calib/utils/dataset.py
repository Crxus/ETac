import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class allData(Dataset):
    def __init__(self, deformation, init_pos, normals) -> None:
        self.deformation = deformation
        self.init_pos = init_pos
        self.normals = normals  # normal vectors at each pts   [n, 3]
    
    def __len__(self):
        return self.deformation.shape[0]
    
    def build_model_input(self, pos, active_deform, normals):
        """
        pos: [n, 3], active_deform: [B, n, 3], normals: [B, n, 3]
        return: [B, n, 9]
        """
        B, n, _ = active_deform.shape
        pos_repeat = pos.unsqueeze(0).expand(B, -1, -1)
        

        return torch.cat([pos_repeat, active_deform, normals], dim=-1)
    
    def build_model_input1(self, pos, active_deform):
        """
        pos: [n, 3], active_deform: [B, n, 3], normals: [B, n, 3]
        return: [B, n, 9]
        """
        B, n, _ = active_deform.shape
        pos_repeat = pos.unsqueeze(0).expand(B, -1, -1)
        

        return torch.cat([pos_repeat, active_deform], dim=-1)

    def prerocess(self, cps):
        '''
        normals [n, 3]
        cps[B]: number of active nodes per env 

        use m points with largest deformation depth to rebuild the deformation field

        return:
        max_cords (B, m, 3)
        min_cords (B, n-m, 3)
        deformation (B, m, 3)
        '''
        b, n, _ = self.deformation.shape

        self.init_points = self.init_pos

        deform = torch.norm(self.deformation, p=2, dim=-1) #[B, n]
        deform_z = self.deformation[:, :, 2]
        self.active_deform = torch.zeros_like(self.deformation)
        self.active_mask = torch.zeros((b, n)).to(self.deformation.device)
        # added
        self.to_active = torch.zeros_like(self.active_mask)  # [B, n], a weight based on distance between each nodes and its nearest active nodes, the closer, the higher
        for i in range(b):
            tpk = cps[i]
            indices = torch.topk(deform_z[i], tpk, largest=False).indices

            if tpk==0:
                pass
            else:
                active_nodes = self.init_points.clone()[indices] # [m, 3] positions of active nodes
                active_dist = torch.norm(
                    self.init_points.unsqueeze(1) - active_nodes.unsqueeze(0), p=2, dim=-1
                ) #[n, m], distance between nodes and active nodes
                min_active_dist = torch.min(active_dist, dim=1).values #[n], minimum distance between each node and active nodes
                mask = min_active_dist!=0
                mid_dist_weight = torch.zeros_like(min_active_dist)
                mid_dist_weight[mask] = torch.max(min_active_dist) / min_active_dist[mask]   
                # [n] the nodes that are close to the active nodes are with high priority, such priority is used for error calculation
                self.to_active[i] = mid_dist_weight / torch.max(mid_dist_weight) # normalize to [0, 1]

            self.active_deform[i, indices] = self.deformation[i, indices]
            self.active_mask[i, indices] = 1  #[b, n]
        self.active_mask = self.active_mask.bool()
        
        self.init_points = self.init_pos

        
        self.init_points = self.init_pos
        res_cords = self.init_points.clone().unsqueeze(0).repeat(b, 1, 1)
        self.ori_max_cords = res_cords
        

        normals_ = self.normals.clone().unsqueeze(0).expand(b, -1, -1)
        mask1 = self.active_mask.clone().unsqueeze(-1).expand(-1, -1, 3)  #[B, n, 3]
        active_normals = torch.zeros_like(normals_)
        active_normals = torch.where(mask1, normals_, active_normals) #[B, n, 3]


        self.data = self.build_model_input(self.init_points, self.active_deform, active_normals)
        #self.data = self.build_model_input1(self.init_points, self.active_deform)
    
    def __getitem__(self, idx):
        # gd->kernel ori_max_cords->kernel init_points->kernel   data -> pointnet
        '''
        active_mask: [B, n] -> bool, which pos are active nodes
        init_points: [B, n, 3]  undeformed positions of nodes
        active_deform: [B, n, 3] displacements of active nodes, for passive nodes, displacements are zero
        data: [B, n, 9] data for network  [undeformed_pos, active displacements, surface normals of active nodes]
        deformation[B, n ,3]: ground truth displacements of all nodes
        to_active[B, n]: a weight based on distance between each nodes and its nearest active nodes, the closer, the higher, used for error calculation
        '''
        return self.active_mask[idx], self.init_points, self.active_deform[idx], self.data[idx], self.deformation[idx], self.to_active[idx]
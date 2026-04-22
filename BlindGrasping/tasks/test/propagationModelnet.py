import torch
import numpy as np
import math
from pytorch3d.ops import sample_farthest_points
import torch.nn.functional as F
import torch.nn as nn

#0731 version

def estimate_normals_pca(points: torch.Tensor, k: int = 20, inner: bool = False) -> torch.Tensor:
    """
    Calculate normals for surface points, using PCA

    parameters:
    - points: [n, 3] points
    - k: number of neighbors, use to conduct a surface
    - inner: adjust the direction of the vectors to the inner sides

    return:
    - normals: [n, 3] 
    """
    n = points.shape[0]
    with torch.no_grad():
        # distances
        dists = torch.cdist(points, points, p=2)

        #  closest k+1 pts (including itself)
        knn_dists, knn_indices = torch.topk(dists, k=k+1, dim=-1, largest=False)

        # get neighbor pts for all pts [n, k+1, 3]
        neighbors = points[knn_indices]  # [n, k+1, 3]

        # centralize
        neighbors_centered = neighbors - neighbors.mean(dim=1, keepdim=True)  # [n, k+1, 3]

        # Calculate the covariance matrix, and use SVD
        cov_matrices = torch.matmul(
            neighbors_centered.transpose(1, 2), neighbors_centered
        ) / k  # [n, 3, 3]

        # Minimum eigenvalue -> eigenvector = normal
        eigvals, eigvecs = torch.linalg.eigh(cov_matrices)  # eigvecs: [n, 3, 3]

        normals = eigvecs[:, :, 0]  # normals [n, 3]
        normals = F.normalize(normals, dim=-1)

        # adjust the direction
        '''center = points.mean(dim=0, keepdim=True)
        outward = points - center  # [n, 3]
        dot_sign = (normals * outward).sum(dim=-1, keepdim=True).sign()
        normals = normals * dot_sign  # default: outside
        if inner:
            normals = normals * (-dot_sign)'''

    return normals

def build_model_input(pos, active_deform, normals):
    """
    pos: [n, 3], active_deform: [B, n, 3], normals: [B, n, 3]
    return: [B, n, 9]
    """
    #B, n, _ = active_deform.shape
    #pos_repeat = pos.unsqueeze(0).expand(B, -1, -1)
    
    return torch.cat([pos, active_deform, normals], dim=-1)

class pnModel(nn.Module):
    def __init__(self, point_feature_dim=64, input_dim = 9, output_dim=3):
        super(pnModel, self).__init__()
        #self.num_points = num_points
        
        # MLP: point features extraction
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            #nn.LayerNorm(64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, point_feature_dim),
            nn.GELU(),
            nn.BatchNorm1d(point_feature_dim)
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(point_feature_dim, point_feature_dim),
            nn.GELU(),
            nn.BatchNorm1d(point_feature_dim),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(2*point_feature_dim, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim),
        )



    def forward(self, x):
        '''
        x: data->[B, n, 9], [undeformed pos, active nodes' displacement(passive nodes with 0 displacement), surface normal of active nodes]
        return: [B, n, 3]
        x1: [B, n, 3], propagated data by kernel
        '''
        b, n, d = x.shape
        origin_data = x.clone() # [B, n, 9]
        x = x.view(b * n, d) 
        x = self.mlp1(x)  #[B*n, point_feature_dim] 

        x = x.view(b, n, -1) # [B, n, point_feature_dim]

        #max pooling
        global_feature = torch.max(x, dim=1)[0] # [B, point_feature_dim]
        global_feature = self.global_mlp(global_feature) 

        global_feature = global_feature.unsqueeze(1).expand(-1, n, -1) 
        new_data = torch.cat((x, global_feature), dim=-1) #[B, n, input_dim+point_feature_dim]

        _, _, d1 = new_data.shape
        new_data = new_data.view(b*n, d1)

        out = self.mlp2(new_data)  # [B*n, 3]
        out = out.view(b, n, -1)  # [B, n, 3]

        return out



class defomationModel(nn.Module):
    def __init__(self, alpha = 50) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha)))  
        self.center = torch.zeros(3)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)
    
    def kernel(self, distances): #order Tylor kernel - exp kernel
        res = torch.exp(-self.alpha*distances)
        return res
    
    
    def get_original_deform(self, points, deform):
        pt_dist = torch.norm(
                points.unsqueeze(2) - points.unsqueeze(1), p=2, dim=-1
            )
        weights = self.kernel(pt_dist)  #[B, m, m]
        return torch.linalg.solve(weights, deform) #tensor [b, m, 3]
    

    def forward(self, centers, points, forces_, direct_deform=True):
        '''
        centers->tensor(B, num_centers, 3)
        points->tensor(B, num_points, 3), active nodes
        forces_->tensor(B, num_points, 3)
        center: geometry center (3)
        return: 
            res->tensor(B, num_centers, 3) = [B, n, 3]
        '''
        forces = None
        if direct_deform == False:
            forces = forces_
            #forces = self.get_original_deform(points, forces_)
        else:
            forces = forces_
        # get the weight: [B, num_centers, num_points]

        # calculate propagation matrix: [B, num_centers, num_pts, 3, 3], a 3*3 matrix for each center pt
        distances = torch.norm(
            points.unsqueeze(1) - centers.unsqueeze(2), p=2, dim=-1
        )  # Shape: (B, num_centers, num_points)
        weights = self.kernel(distances) 
        mid_res = weights.unsqueeze(-1) * forces.unsqueeze(1)  # (B, num_centers, num_points, 3)

        res = torch.sum(mid_res, dim=2)
        return res


class propagationModel:
    def __init__(self, alpha, device):
        self.alpha = alpha
        
        self.deformationModel = defomationModel(alpha)
        self.pn = pnModel().to(device)
    
    def forward(self, centers, points, forces_, contact_normals, center_idx=None, direct_deform=False):
        '''
        centers->tensor(B, num_centers, 3)  [B, n, 3]
        points->tensor(B, num_points, 3), active nodes  [B, m, 3]
        forces_->tensor(B, num_points, 3)   [B, m, 3]
        center_idx [num_centers], indices of centers
        contact_normals: [B, num_points, 3]       
        return: 
            res->tensor(B, num_centers, 3) = [B, n, 3]
        '''

        B, _, _ = points.shape

        forces_mag = torch.norm(forces_, dim=-1)  #[B, n]
        mask = forces_mag != 0  
        active_count = torch.max(torch.sum(mask.int(), dim=-1))
        active_indices = torch.topk(forces_mag, k=active_count, largest=True).indices

        '''forces = None
        if direct_deform==False:
            forces = self.deformationModel.get_original_deform(points, forces_)'''
        forces = forces_

        res = self.deformationModel.forward(centers, points, forces, direct_deform=True)  #[B, num_centers, 3]

        active_normals = torch.zeros_like(contact_normals)
        active_normals[torch.arange(B).unsqueeze(1), active_indices] = contact_normals[torch.arange(B).unsqueeze(1), active_indices]


        '''if B>=1024:
            residual = torch.zeros_like(points)
            batch = int(B/4)
            for i in range(4):
                residual[i*batch: (i+1)*batch] = self.pn(build_model_input(points[i*batch: (i+1)*batch], 
                        forces[i*batch: (i+1)*batch], active_normals[i*batch: (i+1)*batch]))  #[B/4, num_points, 3]
        else:
            residual = self.pn(build_model_input(points, forces, active_normals))  #[B, num_points, 3]'''

        residual = self.pn(build_model_input(points, forces, active_normals))  #[B, num_points, 3]

        residual[torch.arange(B).unsqueeze(1), active_indices] = 0

        if center_idx!=None:
            output = res + residual[:, center_idx]
        else:
            # num_centers = num_pts
            output = res + residual

        return output
        pass


def uni_sample(a, b, n):
    uni_points = np.linspace(0, 1, n, endpoint=False)

    mapped_points = a + (b - a) * uni_points
    sampled_integers = np.round(mapped_points)

    return sampled_integers.tolist()

def space_uni_sample(points, sample_num):
    '''
    points: tensor(n, 3)

    '''
    res = []

    num_points = points.shape[0]
    assert(num_points >= sample_num)

    xy_coords = points[:, :2]
    center = xy_coords.mean(dim=0)

    distances = torch.norm(xy_coords - center, dim=1)
    center_index_in_part = torch.argmin(distances)
    first_point = center_index_in_part.item()

    res += [first_point]
    if sample_num == 1:
        return res

    mask = torch.ones_like(distances, dtype=torch.bool)
    mask[first_point] = False
    remain_points = points[mask]

    remain_xy = remain_points[:, :2]
    _, fps_indices_in_remaining = sample_farthest_points(remain_xy, K=sample_num-1)

    remaining_indices_in_part = torch.arange(num_points)[mask]
    fps_indices_in_part = remaining_indices_in_part[fps_indices_in_remaining]

    res += fps_indices_in_part.cpu().detach().numpy().tolist()

    return res
    pass
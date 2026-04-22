'''
last modified: 2026.1.10
author: Zhe Xu
'''

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, alpha = 50) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha)))  

        self.log_beta = nn.Parameter(torch.log(torch.tensor(10)))  
        self.log_ell = nn.Parameter(torch.log(torch.tensor(10)))  

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)
    
    def kernel(self, distances): #order Tylor kernel - exp kernel
        res = torch.exp(-self.alpha * distances)
        return res
    

    def forward(self, centers, points, displacements):
        '''
        centers->tensor(B, num_centers, 3)
        points->tensor(B, num_points, 3), active nodes
        displacements->tensor(B, num_points, 3)
        center: geometry center (3)
        return: 
            res->tensor(B, num_centers, 3) = [B, n, 3]
        '''
        # get the weight: [B, num_centers, num_points]

        # calculate propagation matrix: [B, num_centers, num_pts, 3, 3], a 3*3 matrix for each center pt
        distances = torch.norm(
            points.unsqueeze(1) - centers.unsqueeze(2), p=2, dim=-1
        )  # Shape: (B, num_centers, num_points)
        weights = self.kernel(distances) 
        mid_res = weights.unsqueeze(-1) * displacements.unsqueeze(1)  # (B, num_centers, num_points, 3)

        res = torch.sum(mid_res, dim=2)
        return res
    
class ResdualNet(nn.Module):
    def __init__(self, point_feature_dim=64, input_dim = 9, output_dim=3):
        super(ResdualNet, self).__init__()
        #self.num_points = num_points
        
        # MLP: point features extraction
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
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

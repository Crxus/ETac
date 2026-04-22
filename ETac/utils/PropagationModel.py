'''
last modified: 2026.1.10
author: Zhe Xu
'''

try:
    from .models import LinearModel, ResdualNet
except ImportError:
    from models import LinearModel, ResdualNet
import torch
import torch.nn.functional as F


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

class propagationModel:
    def __init__(self, alpha=2000, ft_diff = True, device='cuda'):
        # ft_diff: whether use models calibrated from different elastomer to model the displacement propagation for fingertips,
        # since fingertips typically have higher curvature

        self.alpha = alpha
        
        # for the rest part of the hand, we use model calibrated by flat elastomer
        self.linear = LinearModel(alpha).to(device)
        self.net = ResdualNet().to(device)

        #for fingertips, calibrated from biotac
        self.ft_diff = ft_diff
        if self.ft_diff:
            self.ft_linear = LinearModel(alpha).to(device)
            self.ft_net = ResdualNet().to(device)
    
    def load_models(self, rest_ckpts, ft_ckpts=None):
        '''
        rest_ckpts: list, the path of ckpts of linear model as well as the resdual network
        '''
        linear_ckpt, net_ckpt = rest_ckpts
        
        self.linear.load_state_dict(torch.load(linear_ckpt))
        self.net.load_state_dict(torch.load(net_ckpt))

        if self.ft_diff:
            assert ft_ckpts is not None
            ft_linear_ckpt, ft_net_ckpt = ft_ckpts

            self.ft_linear.load_state_dict(torch.load(ft_linear_ckpt))
            self.ft_net.load_state_dict(torch.load(ft_net_ckpt))

    def forward(self, points, displacements_, normals, center_idx=None, fingertip=False):
        '''
        points->tensor(B, n, 3), all nodes on the link
        forces_->tensor(B, n, 3), displacements at all nodes
        center_idx [m], indices of subset nodes
        contact_normals: [B, n, 3] normal directions of surface nodes (undeformed state) 
        return: 
            res->tensor(B, n, 3) = [B, n, 3]
        '''
        B, n, _ = points.shape
        displacement_mag = torch.norm(displacements_, dim=-1)  #[B, n]
        active_mask = displacement_mag != 0      # recognize active nodes, whose displacements are not zero
        no_contact_mask = torch.sum(active_mask.int(), dim=-1) == 0 #[B].  # recognize the no-contact environments

        active_count = torch.max(torch.sum(active_mask.int(), dim=-1))
        active_indices = torch.topk(displacement_mag, k=active_count, largest=True).indices

        displacements = displacements_
        if fingertip:
            res = self.ft_linear.forward(points, points, displacements)  #[B, num_centers, 3]
        else:
            res = self.linear.forward(points, points, displacements)  #[B, num_centers, 3]
        active_normals = torch.zeros_like(normals)
        active_normals[torch.arange(B).unsqueeze(1), active_indices] = normals[torch.arange(B).unsqueeze(1), active_indices]

        data = build_model_input(points, displacements, active_normals)
        if fingertip:
            residual = self.ft_net(data)  #[B, num_points, 3]
        else:
            residual = self.net(data)   #[B, num_points, 3]

        residual = residual  #[B, n, 3] 
        residual[no_contact_mask] = torch.zeros_like(residual[no_contact_mask])
        residual[torch.arange(B).unsqueeze(1), active_indices] = 0

        if center_idx!=None:
            output = res + residual[:, center_idx]
        else:
            # no subset, displacements of all nodes are used as input
            output = res + residual

        return output

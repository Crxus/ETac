import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import os
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, clear_output
import time
import torch.nn.functional as F
import open3d as o3d

def rotate_points_xyz(points, angle_x=0, angle_y=0, angle_z=0, degrees=True):
    """
    apply rotation to the pointcloud, in x → y → z order.
    
    Parameters:
        points: np.ndarray, shape [batch_size, n, 3]
        angle_x, angle_y, angle_z: rotation degree
        degrees: wether use Euler angle, False for radio
    Return:
        pointcloud after rotation, [B, n, 3]
    """
    if degrees:
        angle_x = np.deg2rad(angle_x)
        angle_y = np.deg2rad(angle_y)
        angle_z = np.deg2rad(angle_z)

    # X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x),  np.cos(angle_x)]
    ])

    # Y-axis
    Ry = np.array([
        [ np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])

    # Z-axis
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z),  np.cos(angle_z), 0],
        [0, 0, 1]
    ])

    # Get the rotation matrix
    R = Rz @ Ry @ Rx  # shape [3, 3]
    
    # apply rotation
    rotated = points @ R.T  # [batch_size, n, 3]
    return rotated

def extract_outer_surface(points: np.ndarray, alpha: float = 0.03):
    """
    Extract outer surface from the point cloud. The data collected by us contain the full vertices of the elastomer mesh.

    Extract outer structure using Alpha Shape.
    
    Parameters:
        points: np.ndarray, shape [n, 3], input pointcloud。
        alpha: float, Alpha Shape parameter, smaller alpha leads to a tighter fit, and more points will be considered as outer points. You may need to adjust it based on the scale of your point cloud.
    
    Return:
        outer_mask: np.ndarray, shape [n], boolean array where True indicates points on the outer surface.
    """
    # structure the point cloud using open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # estimate normals, which is required for alpha shape, and adjust the direction to the inner side
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    
    # apply alpha shape to extract the outer surface
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)
    mesh.compute_vertex_normals()
    
    # get the vertices of the mesh, which are the surface points
    surface_points = np.asarray(mesh.vertices)
    surface_indices = set()
    # use cKDTree to find the indices of the surface points in the original point cloud
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(surface_points, k=1)
    surface_indices.update(idx)
    
    # create a boolean mask for the outer points
    outer_mask = np.zeros(points.shape[0], dtype=bool)
    outer_mask[list(surface_indices)] = True
    return outer_mask

def estimate_normals_pca(points: torch.Tensor, k: int = 20, inner: bool = False) -> torch.Tensor:
    """
    Calculate normals for surface nodes, using PCA

    parameters:
    - points: [n, 3] nodes
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

    return normals

def process(deform, ori_pts, curved=True):
    '''
    deform: tensor[B, n, 3]
    ori_pts: tensor[1, n, 3]
    curved: whether the elastomer is curved

    return:
    pts tensor[B, m, 3]: nodes after process
    deformation tensor[B, m, 3]: deformation of the upper surface nodes
    original_pts tensor[1, m, 3]: undeformed upper surface nodes
    '''
    pts = deform + ori_pts
    # rotating is for better visualization, can be ignored it if U do not want
    upper = rotate_points_xyz(pts.clone().cpu().detach().numpy(), angle_x=-90, angle_z=90)
    # upper = pts.clone().cpu().detach().numpy() 

    upper[:, :, 0] -= np.min(upper[:, :, 0])
    upper[:, :, 1] -= np.min(upper[:, :, 1])
    upper[:, :, 2] -= np.min(upper[:, :, 2])
    mask = extract_outer_surface(upper[0], alpha=0.1)
    upper = torch.tensor(upper[:, mask])

    if curved:
        top = int(0.7*upper.shape[1])  
        # exclude the bottom nodes, which are not in contact with the indenter, and thus have no deformation. 
        # U can adjust the ratio to include more or less points.
    else:
        top = upper.shape[1] # flat elastomer
    pts = upper.clone()[:, :top]
    deformation = pts - pts[0].unsqueeze(0)
    # deformation1 = deformation_
    # pts = upper.numpy()

    return pts, deformation, pts[0].unsqueeze(0)
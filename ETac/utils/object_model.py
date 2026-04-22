"""
Last modified date: 2023.02.23
Author: Ruicheng Wang, Jialiang Zhang
Description: Class ObjectModel
"""

'''
object model for tactile information calculation, based on [DexGraspNet](https://pku-epic.github.io/DexGraspNet/)
Last modified date: 2026.01.10
Modified by Zhe Xu, add extra sdf calculation 
'''

import os
import trimesh as tm
import plotly.graph_objects as go
import torch
import pytorch3d.structures
import pytorch3d.ops
import numpy as np

from kaolin.metrics.trianglemesh import compute_sdf

def index_vertices_by_faces(vertices_features, faces):
    r"""Index vertex features to convert per vertex tensor to per vertex per face tensor.

    Args:
        vertices_features (torch.FloatTensor):
            vertices features, of shape
            :math:`(\text{batch_size}, \text{num_points}, \text{knum})`,
            ``knum`` is feature dimension, the features could be xyz position,
            rgb color, or even neural network features.
        faces (torch.LongTensor):
            face index, of shape :math:`(\text{num_faces}, \text{num_vertices})`.
    Returns:
        (torch.FloatTensor):
            the face features, of shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{num_vertices}, \text{knum})`.
    """
    assert vertices_features.ndim == 2, \
        "vertices_features must have 2 dimensions of shape (batch_sizenum_points, knum)"
    assert faces.ndim == 2, "faces must have 2 dimensions of shape (num_faces, num_vertices)"
    input = vertices_features.reshape(-1, 1, 3).expand(-1, faces.shape[-1], -1)
    indices = faces[..., None].expand(
        -1, -1, vertices_features.shape[-1])
    return torch.gather(input=input, index=indices, dim=0)

class ObjectModel:

    def __init__(self, data_root_path, batch_size_each, device="cuda"):
        """
        Create a Object Model
        
        Parameters
        ----------
        data_root_path: str
            directory to object meshes
        batch_size_each: int
            batch size for each objects
        num_samples: int
            numbers of object surface points, sampled with fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path

        self.object_code_list = None
        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_face_verts_list = None
        #self.scale_choice = torch.tensor([0.06, 0.08, 0.1, 0.12, 0.15], dtype=torch.float, device=self.device)
        # we do not apply scale to the object
        self.scale_choice = torch.tensor([1.0], dtype=torch.float, device=self.device) 

    def initialize(self, object_code_list, Selfpath=None):
        """
        Initialize Object Model with list of objects
        
        Choose scales, load meshes, sample surface points
        
        Parameters
        ----------
        object_code_list: list | str
            list of object codes
        """
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        #self.object_scale_tensor = []
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        for object_code in object_code_list:
            #self.scale_choice = torch.tensor([0.06, 0.08, 0.1, 0.12, 0.15], dtype=torch.float, device=self.device)
            #self.object_scale_tensor.append(self.scale_choice[torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each, ), device=self.device)])
            if Selfpath==None:
                meshpath = os.path.join(self.data_root_path, object_code, "coacd")
                temp_mesh = tm.load(os.path.join(self.data_root_path, object_code, "coacd", "decomposed_new1.obj"), force="mesh", process=False)
                self.object_mesh_list.append(temp_mesh)
            else:
                origin_mesh = tm.load(Selfpath, force='mesh')
                self.object_mesh_list.append(origin_mesh)
                
            object_verts = torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device)
            object_faces = torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device)
            for _ in range(self.batch_size_each):
                self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
        #self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)

    def cal_distance(self, x, with_closest_points=False):
        """
        Calculate signed distances from hand contact points to object meshes and return contact normals
        
        Interiors are positive, exteriors are negative
        
        Use our modified Kaolin package
        
        Parameters
        ----------
        x: (B, `n_contact`, 3) torch.Tensor
            hand contact points
        with_closest_points: bool
            whether to return closest points on object meshes
        
        Returns
        -------
        distance: (B, `n_contact`) torch.Tensor
            signed distances from hand contact points to object meshes, inside is positive
        normals: (B, `n_contact`, 3) torch.Tensor
            contact normal vectors defined by gradient
        closest_points: (B, `n_contact`, 3) torch.Tensor
            contact points on object meshes, returned only when `with_closest_points is True`
        """
        n_batches, n_points, _ = x.shape
        distance = []
        normals = []
        closest_points = []
        for i in range(n_batches): #for batch size?
            face_verts = self.object_face_verts_list[i]
            dis, dis_signs, normal, clst_points = compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
            closest_points.append(clst_points)
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        closest_points = torch.stack(closest_points)
        return distance, closest_points, normals

    def get_plotly_data(self, i, color='lightgreen', opacity=0.5, pose=None):
        """
        Get visualization data for plotly.graph_objects
        
        Parameters
        ----------
        i: int
            index of data
        color: str
            color of mesh
        opacity: float
            opacity
        pose: (4, 4) matrix
            homogeneous transformation matrix
        
        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        model_index = i // self.batch_size_each
        mesh = self.object_mesh_list[model_index]
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        data = go.Mesh3d(x=vertices[:, 0],y=vertices[:, 1], z=vertices[:, 2], i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2], color=color, opacity=opacity)
        return [data]
    
    def distances(self, model, ft_contact = False, init = False):
        '''
        calculate the signed distance between hand surface nodes and object mesh

        Parameters:
        model: hand model, provides surface nodes after kinematics
        ft_contact: bool, whether to only consider fingertips tactile sensing
        init: bool, whether it is the first time to calculate the distance


        return:
        dis: tensor[B, n], signed distances between hand surface nodes and object mesh, positive is inside -> penetration
        clst: tensor[B, n, 3], closest points on object mesh for each surface node
        surface: tensor[B, n, 3], surface nodes of the entire hand
        norm: tensor[B, n, 3], direction vector from surface nodes to their closest points on object mesh, normalized 
        (for active nodes, such direction is the contact normal)

        Since take the displacements of all surface nodes as input for the policy may be redundant, especially when the number of surface nodes is large.
        Therefore, ETac supports to uniformly select a subset of surface nodes, only input their displacements to the policy.
        
        Surface nodes cover the entire hand. But in grasping task, we only care about the nodes on the side of palm, 
        which are more likely to be in contact with the object and provide useful tactile information.

        At such circumstances, if init is True, also return:
        cord: tensor[n//2], indices of the nodes on the palm side
        cord_dict: dict, indicates the links that the nodes on the palm side belong to.


        cord: tensor[m], indices of the selected subset of surface nodes.
        cord_dict: dict, indicates the links that the selected surface nodes belong to.
        '''

        fingertips = ["robot0:ffdistal_child", "robot0:mfdistal_child", "robot0:rfdistal_child", "robot0:lfdistal_child", "robot0:thdistal_child"] #for shadow
        except_link=['robot0:ffknuckle_child',
                    'robot0:mfknuckle_child',
                    'robot0:rfknuckle_child',
                    'robot0:lfknuckle_child',
                    'robot0:thhub_child',
                    'robot0:thbase_child']  # For ShadowHand, these links do not appear on the surface of the hand, so we do not sample surface nodes on these links. 
        # if U are using other hand models, such as Allegro or LeapHand, please check the link names and modify, according to the urdf files.
        hand_model=model
        dis = torch.tensor([]).to(self.device)
        clst = torch.tensor([]).to(self.device)
        norm = torch.tensor([]).to(self.device)
        surface = torch.tensor([]).to(self.device)
        cord = torch.tensor([]).to(self.device)
        cord_dict = {}
        mark = 0
        for i, link_name in enumerate(hand_model.mesh):
            if hand_model.mesh[link_name]['surface_points'].shape[0] == 0:
                continue
            if link_name in except_link:
                continue
            if ft_contact:
                if link_name not in fingertips:
                    continue
            surface_points = hand_model.current_status[link_name].transform_points(
                hand_model.mesh[link_name]['surface_points']).expand(self.batch_size_each, -1, 3)
                
            surface_points = surface_points @ hand_model.global_rotation.transpose(
                1, 2) + hand_model.global_translation.unsqueeze(1)

            distances, clst_points, normals = self.cal_distance(
                surface_points)
            norm =torch.cat((norm, normals), dim = 1)
            dis = torch.cat((dis, distances), dim = 1)
            clst = torch.cat((clst, clst_points), dim = 1)
            surface = torch.cat((surface, surface_points), dim = 1)
            if init:
                temp_points = surface_points[0] # at the init phase, the surface nodes are not transformed
                # get the nodes on the palm side
                cord_ind = temp_points[:, 2].topk(temp_points.shape[0]//2, largest = False).indices + mark
                cord_ind, _ = torch.sort(cord_ind)
                #cord_points = temp_points[cord_ind]
                cord = torch.cat((cord, cord_ind.float()), dim = 0)
                cord_list = cord_ind.cpu().numpy().tolist()
                cord_dict[link_name] = cord_list
                mark += surface_points.shape[1]
        
        if init:
            return dis, clst, surface, norm, cord.long(), cord_dict

        #return dis, clst, surface
        return dis, clst, surface, norm

    def kinematics(self, translation, rotation):
        '''
        apply kinematics to the object mesh
        input:
        rotation: torch.Size([B, 3, 3])  // for batch cases replace '1' to 'batch_size'
        translation: torch.Size([B, 3])

        change the vertices and object_face_verts_list
        '''
        #translation.to(self.device)
        for index in range(len(self.object_mesh_list)): # len(self.object_mesh_list) = 0
            verts = torch.Tensor(self.object_mesh_list[index].vertices)
            faces = torch.Tensor(self.object_mesh_list[index].faces).long()

            verts = verts @ rotation.transpose(1, 2) + translation.unsqueeze(1)
            '''self.object_mesh_list[index] = tm.Trimesh(vertices = verts.detach()[0], 
                                                      faces = faces.cpu())'''
            for i in range(len(verts)):
                self.object_face_verts_list[i] = index_vertices_by_faces(verts[i].detach(), 
                                                    faces).to(self.device)
    

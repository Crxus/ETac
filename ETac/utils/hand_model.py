"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: Class HandModelMJCFLite
"""

'''
hand model for tactile information calculation, based on [DexGraspNet](https://pku-epic.github.io/DexGraspNet/)
'''

import os
import torch
# from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from .rot6d import robust_compute_rotation_matrix_from_ortho6d
import pytorch_kinematics as pk
import trimesh
import pytorch3d.structures
import pytorch3d.ops


class HandModelMJCFLite:
    def __init__(self, mjcf_path, mesh_path=None, device='cpu', n_surface_points = 0):
        """
        Create a Lite Hand Model for a MJCF robot
        
        Parameters
        ----------
        mjcf_path: str
            path to mjcf file
        mesh_path: str
            path to mesh directory
        device: str | torch.Device
            device for torch tensors
        """
        self.device = device
        self.chain = pk.build_chain_from_mjcf(
            open(mjcf_path).read()).to(dtype=torch.float, device=device)
        # self.chain = pk.build_chain_from_urdf(
        #     open(mjcf_path).read()).to(dtype=torch.float, device=device)  for urdf file
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        self.mesh = {}
        areas = {}

        def build_mesh_recurse(body):
            if (len(body.link.visuals) > 0):
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    
                    scale = torch.tensor(
                        [1, 1, 1], dtype=torch.float, device=device)
                    if visual.geom_type == "box":
                        link_mesh = trimesh.primitives.Box(
                            extents=2*visual.geom_param)
                    elif visual.geom_type == "capsule":
                        link_mesh = trimesh.primitives.Capsule(
                            radius=visual.geom_param[0], height=visual.geom_param[1]*2).apply_translation((0, 0, -visual.geom_param[1]))
                    else:
                        link_mesh = trimesh.load_mesh(
                            os.path.join(mesh_path, visual.geom_param[0].split(":")[1]+".obj"), process=False)
                        if visual.geom_param[1] is not None:
                            scale = (visual.geom_param[1]).to(dtype=torch.float, device=device)
                    vertices = torch.tensor(
                        link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(
                        link_mesh.faces, dtype=torch.float, device=device)
                    pos = visual.offset.to(dtype=torch.float, device=device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                self.mesh[body.link.name] = {'vertices': link_vertices,
                                             'faces': link_faces,
                                             }
                areas[body.link.name] = trimesh.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()
            for children in body.children:
                build_mesh_recurse(children)
        build_mesh_recurse(self.chain._root)

        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(body.joint.range[0])
                self.joints_upper.append(body.joint.range[1])
            for children in body.children:
                set_joint_range_recurse(children)
        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.stack(
            self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(
            self.joints_upper).float().to(device)

        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])  
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values())  #list(num_samples.keys())[0] --> palm?
        for link_name in self.mesh:
            if num_samples[link_name] == 0:  #there will not be sample points
                self.mesh[link_name]['surface_points'] = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                continue
            #build up a mesh
            mesh = pytorch3d.structures.Meshes(self.mesh[link_name]['vertices'].unsqueeze(0), self.mesh[link_name]['faces'].unsqueeze(0))
            #build point cloud
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            #sample_farthest_points: sample points, the new sampled point is fartherest from any sampled points
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]['surface_points'] = surface_points

        self.hand_pose = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None

    def set_parameters(self, hand_pose):
        """
        Set translation, rotation, and joint angles of grasps
        
        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
            self.hand_pose[:, 3:9])
        self.current_status = self.chain.forward_kinematics(
            self.hand_pose[:, 9:])

    def get_trimesh_data(self, i):
        """
        Get full mesh
        
        Returns
        -------
        data: trimesh.Trimesh
        """
        data = trimesh.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            data += trimesh.Trimesh(vertices=v, faces=f)
        return data

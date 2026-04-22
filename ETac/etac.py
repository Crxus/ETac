'''
    last modified: 2026.1.10
    author: Zhe Xu

    what do u need: hand urdf file path, hand mesh path, object mesh path, 
    the path of the .txt file which saves the cordinates of tactile points, the file saves fsr/taxel indices

    to get tactile info u need: hand state ([B, 7 + num_dofs], translation & rotation & dof states), object state ([B, 7], translation & rotation), 
    states of each link of the hand ([B, 7], translation & rotation)

    sequence: init -> setup_models(once) -> cal_tactile_info

    in instance:

    initialize:
    self.model = models(hand_file_path, hand_mesh_path, object_root_path, device, num_points, batch_size)
    self.model.setup_models('pts/leappts-4000.txt', 'pts/leapfsr-4000.txt', 'pts/leaptaxel-4000.txt')
    self.model.set_object(mesh_file_path)

    get tactile info(durint simulation):
    fsr_vec, taxel_vec = model.cal_tactile_info(hand_state, object_state, all_link_states)
'''

from .utils.hand_model import HandModelMJCFLite
from .utils.object_model import ObjectModel
from .utils.PropagationModel import propagationModel, estimate_normals_pca
import torch
import numpy as np
import transforms3d

class ETac:
    def __init__(self, hand_file_path, hand_mesh_path, object_root_path, num_points, batch_size, device):
        '''
        hand_file_path: str, path to the hand urdf/mjcf file
        hand_mesh_path: str, path to the hand mesh file
        object_root_path: str, path to the object root directory
        device: torch.device, device to run the model on
        num_points: int, number of surface nodes on the hand
        batch_size: int, size of the batch
        '''
        # setup models for contact detection & displacement calculation
        self.hand_model = HandModelMJCFLite(mjcf_path=hand_file_path,
                                            mesh_path=hand_mesh_path, 
                                            n_surface_points=num_points,
                                            device=device)
        self.object_model = ObjectModel(data_root_path=object_root_path, 
                                        batch_size_each=batch_size, 
                                        device=device)
        self.device = device
        
        self.fingertips = ["robot0:ffdistal_child", "robot0:mfdistal_child", "robot0:rfdistal_child", "robot0:lfdistal_child", "robot0:thdistal_child"] #for shadow
        self.except_link=['robot0:ffknuckle_child',
                    'robot0:mfknuckle_child',
                    'robot0:rfknuckle_child',
                    'robot0:lfknuckle_child',
                    'robot0:thhub_child',
                    'robot0:thbase_child']  
        # For ShadowHand, these links do not appear on the surface of the hand, so we do not sample surface nodes on these links. 
        # if U are using other hand models, such as Allegro or LeapHand, please check the link names and modify, according to the urdf files.

        self.propagation_model = propagationModel()

        # hyper parameters of the contact model
        '''
        According to Efficient Tactile Simulation with Differentiability for Robotic Manipulation (https://tactilesim.csail.mit.edu/), 
        each parameter belongs a different range:

            k_n: [50, 450]
            k_t: [0.2, 2.3]
            mu: [0.5, 2.5]
            k_d: [0, 100]
        '''
        self.k_n = 150
        self.k_d = 37
        self.k_t = 1.1
        self.mu = 1.3

        self.init = False

    def setup_propagation(self, models_paths, ft_models_paths):
        '''
        models_paths: list of pretrained checkpoints paths of linear model & resdual net, same as ft_models_paths
        '''
        self.propagation_model.load_models(models_paths, ft_models_paths)

    def set_dicts(self):
        '''
        ind2link: dict, indicates which nodes belong to a certain link
        '''
        self.ind2link = {}
        self.num_links = 0
        mark=0
        for link in self.hand_model.mesh:
            if link in self.except_link:
                continue
            self.ind2link[link] = [mark, mark + self.hand_model.mesh[link]['surface_points'].shape[0]]
            mark+=self.hand_model.mesh[link]['surface_points'].shape[0]
            self.num_links+=1 # if only use fingertip tactile info, num_link = 5

    def set_object(self, object_mesh_path):
        self.object_model.initialize(object_mesh_path)
        pass

    def set_hand_model_pos(self, hand_pos, hand_dofs):
        '''
        hand_pos: tensor[B, 7], hand root trans+rotations 
        hand_dof: hand dofs [B, num_dofs]
        '''
        hand_trans = hand_pos[:, :3]
        hand_rots = hand_pos[:, 3:7]

        mid_dof_states = hand_dofs.clone().cpu().numpy().tolist()
        index=[3,0,1,2]  # the rotation from Isaac Gym needs reorder, please modify it if U use other simulation engine 
        mid_rotat = hand_rots[:, index].clone().cpu().numpy().tolist()

        for ind in range(len(mid_rotat)):
            rot=np.array(transforms3d.euler.quat2mat(mid_rotat[ind]))
            rot= rot[:, :2].T.ravel().tolist()
            mid_rotat[ind] = rot

        global_translation=hand_trans.cpu().numpy().tolist()
        temp = np.concatenate((global_translation, mid_rotat, mid_dof_states), axis=1)

        hand_pose=torch.tensor(temp, dtype=torch.float, 
                                device = self.device)
        hand_state = hand_pose.to(self.device).requires_grad_()
        self.hand_model.set_parameters(hand_state) # 3-trasnlation, 9-flatten rotation matrix, 16-dof

    def set_object_model_pos(self, object_pos):
        '''
        object_pos: tensor[B, 7], object trans+rotations
        '''
        obj_trans = object_pos[:, :3]
        obj_rots = object_pos[:, 3:7]

        object_translation = obj_trans
        index=[3,0,1,2]
        object_rotation = obj_rots[:, index].clone().cpu().numpy().tolist()

        for ind in range(len(object_rotation)):
            object_rotation[ind] = (np.array(transforms3d.euler.quat2mat(object_rotation[ind]))).tolist()

        object_rotation = torch.Tensor(object_rotation)
        self.object_model.kinematics(object_translation.cpu(), object_rotation.cpu())

        '''
    TODO: calculate the velcities of surface nodes, we calculate link by link
    '''
    def cal_vel(self, root_trans, points, vel, angvel):
        '''
        input:
            root_trans-> (B, 3): the position of the link root
            points-> (B, num_points, 3): nodes belong to the link
            vel-> (B, 3): velocity of the link root
            angvel-> (B, 3): angle velocity of the link root
        return:
            velocity -> (B, num_points), velocity of the points
        '''
        pos_delta = points - root_trans.unsqueeze(1) # (B, num_points, 3), relative pos
        velocity = vel.unsqueeze(1) + torch.cross(pos_delta, angvel.unsqueeze(1), dim = -1)

        return velocity


    def cal_force(self, depths, normal_directions, vel_delta):
        '''
        depth->[B, n]: the penetration depth at each node, for the node that does not contact with the obj, its depth is 0
        normal_direction->[B, n, 3]: the direction vector of the contact normals
        vel_delta->[B, n, 3]: the relative vel of each node (w.r.t. the clst points of each of them)

        return:
            contact forces -> [B, n, 3]
        '''
        if (depths!=0).any() == False:
            return torch.zeros_like(normal_directions)
        v_n = torch.sum(normal_directions * vel_delta, dim = -1).unsqueeze(-1) * normal_directions # get the velocity at normal direction
        temp = (-self.k_n + torch.norm(v_n, p = 2, dim = -1) * self.k_d) # tensor(B, n)
        f_n = temp.unsqueeze(-1) * normal_directions * depths.unsqueeze(-1) # (B, n, 3)

        # v_n_ = torch.sum(normal_directions * vel_delta, dim = -1).unsqueeze(-1)
        # v_n = v_n_ * normal_directions # get the velocity at normal direction
        # #temp = (-self.k_n + torch.norm(v_n, p = 2, dim = -1) * self.k_d) # tensor(B, num_points)
        # temp = (-self.k_n + v_n_ * self.k_d)
        # f_n = temp * normal_directions * depths.unsqueeze(-1) # (B, num_points, 3)
        

        v_t = vel_delta - v_n # get the velocity at tangential direction
        v_t_ = torch.norm(v_t, p = 2, dim = -1)  # (B, n)

        temp_vt = self.k_t * v_t_
        temp_fn = self.mu * torch.norm(f_n, p = 2, dim = -1)
        f_t = -v_t/(v_t_.unsqueeze(-1)) * torch.min(temp_vt, temp_fn).unsqueeze(-1) # (B, n, 3)

        contact_forces = f_n + f_t

        return contact_forces

    def cal_tactile_info(self, hand_state, object_state, dof_pos, all_link_states, ft_contact=False):
        '''
        hand_state: [B, 7], pos + rot + vel + angvel
        object_state: [B, 13], same
        dof_pos: [B, 16], positions of all dofs
        
        all_link_states: [B, num_links, 13] states of all hand links

        return:
        cps: tensor[B], number of active nodes (nodes in contact)
        contact_vector: tensor[B, num_links], one-hot vector indicating which link of hand is in contact
        displacements: tensor[B, num_points, 3], the propagated displacements of each node
        '''
        B, _ = hand_state.shape

        hand_pos = hand_state[:, :7]

        object_pos = object_state[:, :7]
        object_trans = object_pos[:, :3]
        object_vel = object_state[:, 7:10]
        object_angvel = object_state[:, 10:13]

        all_link_trans = all_link_states[:, :, :3]
        all_link_vels = all_link_states[:, :, 7:]

        self.set_hand_model_pos(hand_pos, dof_pos)
        self.set_object_model_pos(object_pos)

        dist = None  # [B, num_points], signed distances between each node and the object mesh
        clst_points = None # closest points on the object mesh w.r.t each node
        surface_points = None # nodes after fk
        norm = None # contact normals

        if self.init == False:
            self.init = True
            
            # self.cord: index of the nodes on the palm side
            dist, clst_points, surface_points, norm = self.object_model.distances(self.hand_model, ft_contact)

            self.origin_points = surface_points # undeformed surface nodes, just need to be assigned once

            # surface normals of each nodes at undeformed state, assign once
            self.normals = {}
            for link in self.ind2link.keys():
                lower = self.ind2link[link][0]
                upper = self.ind2link[link][1]
                pts = self.origin_points[0, lower: upper] # [n, 3]
                center_pt = torch.stack([torch.mean(pts[:, 0]), torch.mean(pts[:, 1]), torch.mean(pts[:, 2])])
                k = 20
                if (upper - lower)<k:
                    k = (upper - lower)-1
                ori_normals = estimate_normals_pca(pts, k=k, inner=True) #[n, 3]
                center_direct = center_pt.unsqueeze(0) - pts
                flip_mask = (center_direct * ori_normals).sum(dim=-1) < 0
                ori_normals[flip_mask] = -ori_normals[flip_mask]
                self.normals[link] = ori_normals.to(self.device)

        else:
            dist, clst_points, surface_points, norm = self.object_model.distances(self.hand_model)

        thre = 0.002 # 2mm as contact threshold, modify it if U want

        # indicate which nodes are in contact
        mask = (dist < thre) & (dist >= 0)  # positive distance means direct penetration
        mask_ = mask | ((-dist < thre) & (dist < 0))  # [B, num_points], we slightly expand the sensing range to enlarge the number of active (contact) nodes

        # penetration depths
        depths = dist.clone() + thre # dist -> [-0.002, 0.002], depths -> [0, 0.004]
        # avoid divide zero
        depths = torch.where(mask_.int()==1, torch.where(depths==0, torch.zeros_like(depths)+1e-4, torch.abs(depths)), torch.zeros_like(depths))
        forces_ = torch.zeros(B, surface_points.shape[1], 3).to(self.device) #[B, num_points, 3], forces in 3-axis

        # cps: number of active nodes, indicating the contact area
        cps = torch.sum(mask_.int(), dim = -1) #[B]

        # set the one-hot vector S_cr, which indicates which link is in contact
        contact_vector = torch.zeros(B, self.num_links).to(self.device) # [B, num_links]

        for i, mas in enumerate(mask_):
            if cps[i]==0:
                continue
            indices = torch.nonzero(mas)
            j = 0
            for link in self.ind2link.keys():
                if link in self.except_link:
                    continue
                lower = self.ind2link[link][0]
                upper = self.ind2link[link][1]
                con1 = indices >= lower
                con2 = indices < upper
                if (con1 & con2).any():
                    contact_vector[i, j]=1 # in the i-th environment, the j-th link contacts with the obj
                j += 1

        # get the velocities of the cloest points (of each surface node) on the object mesh w.r.t. the object root
        obj_vel = self.cal_vel(object_trans, clst_points, object_vel, object_angvel) # [B, n, 3]

        # calculate the forces/displacements at active nodes
        displacements = torch.zeros_like(surface_points) # [B, n, 3], the propagated displacements of each node, which is also the tactile reading we want to get
        for i, link in enumerate(self.ind2link.keys()):
            
            lower = self.ind2link[link][0]
            upper = self.ind2link[link][1] # for each link, get the nodes belong to it
            
            link_trans = all_link_trans[:, i].squeeze()
            linvel = all_link_vels[:, i, :3].squeeze()
            angvel = all_link_vels[:, i, 3:6].squeeze()
            
            with torch.no_grad():
                # calculate the exact velocity of each node
                vel = self.cal_vel(link_trans, surface_points[:, lower: upper], linvel, angvel)
                # relative velocity between each node and the object mesh
                delta_vel = vel - obj_vel[:, lower: upper] 
                # forces [B, num_points], magnitude of the force
                # force_ [B, num_points, 3]
                contact_force = self.cal_force(depths[:, lower: upper], norm[:, lower: upper], delta_vel)
                forces_[:, lower: upper] = contact_force

                if link in self.fingertips:
                    link_normals = (self.normals[link].clone()).unsqueeze(0).expand(B, -1, -1)
                    displacements[:, lower: upper] = self.propagation_model.forward(self.origin_points[:, lower: upper], 
                                                                                    forces_[:, lower: upper], link_normals, fingertip=True) 
                else:
                    # in our case, propagation model of non-fingertip areas is calibrated using the flat elastomer
                    link_normals = torch.zeros_like(forces_[:, lower: upper])
                    link_normals[:, :, 2] = -1
                    displacements[:, lower: upper] = self.propagation_model.forward(self.origin_points[:, lower: upper], 
                                                                                    forces_[:, lower: upper], link_normals) 
                    
        return cps.squeeze(), contact_vector, displacements
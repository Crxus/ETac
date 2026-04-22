# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Last modified: 2026.1.10
Author: Zhe Xu
Description: Blind Grasping environment with in Isaac Gym.
'''

import numpy as np
import os
import torch, gc
import sys

from rl_games.algos_torch import torch_ext

# from ETac.etac import ETac
try:
    from ETac import ETac
except ModuleNotFoundError:
    # 回退：BlindGrasping 与 ETac 为并列目录时，加入工作区根目录到 sys.path
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _workspace_root = os.path.dirname(_project_root)
    if _workspace_root not in sys.path:
        sys.path.insert(0, _workspace_root)
    from ETac.etac import ETac

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil


from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp  

from isaacgymenvs.tasks.base.vec_task import VecTask

from utils.hand_model_lite_new import HandModelMJCFLite

import math

import json

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class ShadowHand(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        #option: store success rate
        self.succ_rates = torch.tensor([], device=rl_device)
        self.succ_count = 0
        # count the finished step
        self.count = 0
        self.succ_count = 0
        self.episode_buffer = torch.tensor([], device=rl_device)
        self.stab_buffer = torch.tensor([], device=rl_device)

        self.cfg = cfg

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]

        # added
        self.out_dist = self.cfg["env"]["outDistance"] #0.15
        self.fall_penalty = self.cfg["env"]["fallPenalty"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        #position
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        #force
        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)
        # speed scale? 
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        # trans (force) & orientation scale (torque), corresponding to the action
        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        #exp setup: episode, reset time, etc.
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        # averange factor or sth?
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]



        self.asset_codes = {
            'apple': "sem-Fruit-b095a1c41447e9da887d2d6e4a6617b3", 
            'orange': "sem-Fruit-cdaccd0978028eef6c80f6a5291c505",
            'banana': "sem-Fruit-5b102f0f4ceccf3aec8b4278552324cf",
            'clock': "sem-TableClock-8ac6b93f8d93c400923ef79fcb120ce8",
            'soap': "sem-SoapBar-8da3c818adf836fcf2c4595240954986",
            'radio': "sem-Radio-7d1f7561729128f41f8d5edb17ddb332",
            'rock': "sem-Rock-386c91ddd48b956f85b5e1f7eb9f2b9c",
            'bowl': "sem-Bowl-2eb4cfc59205bb3a147c505998f546dd",
            'sodacan': "sem-SodaCan-1b36df7ed7ddd974c538fbfc1e40dbe0",
            'tank': "sem-Tank-cca0b9a74f1e341386459c315bdea98e", 
            'teapot': "sem-Teapot-5a471458da447600fea9cf313bd7758b",
            'battery': "sem-Battery-7ecc6a361de0128ec8de11d571206924",
            'food': "sem-FoodItem-1a10ecfcaac04c882d17d82c03b66",
            'light': "sem-LightBulb-f16f59cbfaa7de05ff2eead8e091e281",
            'camera': "sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5", 
            'flash': "sem-Flashlight-70cfc22ffe0e9835fc3223d52c1f21a9",
            'egg': 'egg',
            'cube': 'cube', 
        }

        assert self.object_type in self.asset_codes.keys()

        self.iniflag=False  #when it is False, means the object is in initial state



        self.start_heights = {
            'block': 0.06,
            'sodacan': 0.05,
            'banana': 0.05,
            'teapot': 0.1,
            'tank': 0.08,
            'rock': 0.08,
            'egg': 0.07,
            'food': 0.06,
            'apple': 0.06,
        }

        self.asset_meshes_dict = {
            "block": "./data/meshdata/cube_multicube/meshes/dex_cube.obj",
            "ball": './data/meshdata/cube_multicube/meshes/ball.obj'
        }

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["blind"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [blind]")

        print("Obs type:", self.obs_type)

        #try to add new obs, contact vector or num of contact points?
        self.num_obs_dict = {
            "blind": 135,
        }

        self.up_axis = 'z'
        
        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.fts = ["robot0:ffdistal_child", "robot0:mfdistal_child", "robot0:rfdistal_child", "robot0:lfdistal_child", "robot0:thdistal_child"]
        except_link=['robot0:ffknuckle_child',
                    'robot0:mfknuckle_child',
                    'robot0:rfknuckle_child',
                    'robot0:lfknuckle_child',
                    'robot0:thhub_child',
                    'robot0:thbase_child']


        self.num_fingertips = len(self.fingertips)

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        self.contact_obs = self.cfg["env"]["contact_observations"]
        self.num_surface_points = self.cfg["env"]["numSurfacePoints"]
        self.ft_contact = self.cfg["env"]["fingerTips"]
        self.blind = self.cfg["env"]["blind"]
        self.num_tac_features = self.cfg["env"]["subFeatures"]
        self.feature_factor = self.num_surface_points/2000


        self.etac = ETac(hand_file_path='mjcf/shadow_hand_vis.xml', 
                         hand_mesh_path='mjcf/meshes', 
                         object_root_path="./data/meshdata", 
                         num_points=self.num_surface_points,
                         batch_size=self.cfg["env"]["numEnvs"], 
                         device=rl_device)
        
        with open('./points-1000-0903.txt', 'r') as f:
            crap = json.loads(f.read())
            for link in crap.keys():
                self.etac.hand_model.mesh[link]['surface_points'] = torch.tensor(crap[link]).to(rl_device)
        self.etac.set_dicts()

        if self.ft_contact: # check the number of nodes
            temp=0
            for link in self.etac.hand_model.mesh:
                if link not in self.fts:
                    continue
                temp+=len(self.etac.hand_model.mesh[link]['surface_points'])
            assert(temp <= self.num_surface_points)
            self.num_surface_points = temp
        else:
            temp=0
            for link in self.etac.hand_model.mesh:
                if link in except_link:
                    continue
                temp+=len(self.etac.hand_model.mesh[link]['surface_points'])
            assert(temp <= self.num_surface_points)
            if self.num_tac_features == self.num_surface_points:
                self.num_tac_features = temp
            self.num_surface_points = temp

        # initialize the object 
        self.etac.set_object(self.asset_codes[self.object_type])

        num_states = 0
        if self.asymmetric_obs:
            num_states = 174 
            if self.ft_contact: # checked
                num_states = 184

        

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        if self.obs_type == "blind" and self.contact_obs == False:
            self.cfg["env"]["numObservations"] = 198
        if self.contact_obs == True and self.ft_contact==True: # checked
            self.cfg["env"]["numObservations"] = 174

        self.cfg["env"]["numStates"] = num_states
        # To apply the kinematics of the whole hand (i.e., hand root), do we need to add number of actions?  maybe 6? a force and torque?
        # 18 acurators
        self.cfg["env"]["numActions"] = 24

        self.cfg["env"]["numObservations"] += 3*self.num_tac_features  
        self.cfg["env"]["numStates"] += 3*self.num_tac_features # 3-diminsion displacement for each node


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, 
                         graphics_device_id=graphics_device_id, headless=headless, 
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # add for log
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "ft_dist",
                "hand_dist",
                "goal_hand_rew",
                "lift_rew",
                "con_rew",
                "link_rew",
            ]
        }

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


        # get gym GPU state tensors: root, dofs and each links(rigid body)
        # get hands and objs root states 4 all envs  3 actor each env, 1 hand & 2 objs (obj_handle, goal_handel)
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # dof state contains pos and vel
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # same as root_state 4 rigid body
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        # add part, save the init state 4 reset
        self.init_root_tensor = self.root_state_tensor.clone()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        # target pose 4 previous step & current step
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.pose_diff = torch.zeros_like(self.reset_buf).to(self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        #average factor for success rate calculation
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # apply force and torque to hand
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)


        # parameters for the contact model
        '''
        According to Efficient Tactile Simulation with Differentiability for Robotic Manipulation, 
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

    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        #y for 2-D panel and z for 3-D
        self.up_axis_idx = 2 if self.up_axis == 'z' else 1 # index of up axis: Y=1, Z=2
        self.sim_params.physx.contact_offset = 0.005
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
            #pass

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # range of exp region?
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/meshdata'))
        shadow_hand_asset_file = os.path.normpath("mjcf/open_ai_assets/hand/shadow_hand.xml")

        if "asset" in self.cfg["env"]:
            # asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = os.path.normpath(self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file))

        object_asset_file = self.asset_codes[self.object_type] + "/coacd/coacd1.urdf"

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)
        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    #    Damping to be used on attraction solver.
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]

        # get shadow_hand dof properties, loaded by Isaac Gym from the MJCF file
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        #change dof properties to torch
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        #get figer tip handles
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        #added get all link handles
        except_link=['robot0:ffknuckle_child',
                    'robot0:mfknuckle_child',
                    'robot0:rfknuckle_child',
                    'robot0:lfknuckle_child',
                    'robot0:thhub_child',
                    'robot0:thbase_child']

        links = []
        for link in self.etac.hand_model.mesh:
            if link in except_link:
                pass
                #continue
            if link == 'robot0:palm':
                links.append(link)
            else:
                links.append(link[: -6])
        self.link_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in links]
        # added end


        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = False
        object_asset_options.density=567.0
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        #As for goal asset, do not consider gravity
        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        shadow_hand_start_pose = gymapi.Transform()
        #initial position
        #up_axis_idx = 2 when creating a 3-d panel
        init_hei = 0.08
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(init_hei, self.up_axis_idx))
        #shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.7, self.up_axis_idx))
        #return [0,0,0.5] in this case?

        #shadow_hand_start_pose.r = gymapi.Quat()11hand_start_pose
        shadow_hand_start_pose.r.x = 0.3 * math.pi    


        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = shadow_hand_start_pose.p.x
        #to let the initial position be in the center of the palm?
        # we set the dy range to [-0.2, 0.0]
        # we set the dx range to [-0.06, 0.06]
        pose_dy, pose_dz = -0.12, -0.1
        #pose_dy, pose_dz = -0.12, -0.4
        self.dy = pose_dy

        object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        object_start_pose.r.z = pose_dy

        #object_start_pose.r.z = pose_dy     

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.40)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)

        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + 2  #asset_rigid_body_count
        max_agg_shapes = self.num_shadow_hand_shapes + 2  #asset_rigid_shape_count

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []
        self.goal_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        roots=['robot0:palm']  # root link for ShadowHand, change it if U r using a different hand model.
        obj_name = [self.object_type]
        self.roots=[self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in roots]

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        self.objs=[self.gym.find_asset_rigid_body_index(object_asset, name) + 
                  self.gym.get_asset_rigid_body_count(shadow_hand_asset)
                   for name in obj_name]
        

        shadow_hand_rb_count = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        #get the index of the object rigid bodys
        self.object_rb_handles = list(range(shadow_hand_rb_count, shadow_hand_rb_count + object_rb_count))

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM) #get hand id
            # get the index of hands with respect to the whole simulation
            self.hand_indices.append(hand_idx)

            # enable DOF force sensors, if needed
            if self.obs_type == "full_state" or self.asymmetric_obs or self.obs_type == "blind":
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor) 
            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)

            object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            
            
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_states.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                     goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z, goal_start_pose.r.w,
                                     0, 0, 0, 0, 0, 0])
            self.goal_object_indices.append(goal_object_idx)
            # apply scale for object from ShapeNet, not sure
            self.gym.set_actor_scale(env_ptr, goal_handle, 0.001)
            if self.object_type in self.asset_codes.keys():
                self.gym.set_actor_scale(env_ptr, goal_handle, 0.001)

            # if self.object_type != "cube":
            self.gym.set_rigid_body_color(
                env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        # we are not using new mass values after DR (domain randomization?) when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = to_torch(self.goal_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        ################################################################################
        #### TODO: modify goal state
        ###################################################################################
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.dof_pos = self.shadow_hand_dof_pos

        self.rewards, self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:] = compute_grasp_reward(
                self.hand_pos, self.object_pos, self.num_surface_points, self.cps, self.contact_vector, self.fingertip_pos,
                self.reset_buf, self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
                self.max_episode_length, self.goal_pos, 
                self.actions, self.out_dist, self.av_factor
            )
        
        for key, value in self.rewards.items():
            self._episode_sums[key] += value

        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))
    # used for the input of rl
    def compute_observations(self):
        #basic: dof state(pos, rot), root state, rigid body state
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if (self.obs_type == "full_state" or self.asymmetric_obs) and self.obs_type!= "blind":
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.hand_pose = self.root_state_tensor[self.hand_indices, 0:7]
        self.hand_pos = self.root_state_tensor[self.hand_indices, 0:3]
        self.hand_rot = self.root_state_tensor[self.hand_indices, 3:7]

        #state and pos 4 all 5 finger tips
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        self.cps, self.contact_vector, self.displacements  = self.setup_model()

        if self.obs_type == "blind":
            self.compute_blind(w_contact=self.contact_obs)
        else:
            print("Unknown observations type!")

        if self.asymmetric_obs:
            self.compute_blind(asymm_obs=True)
            

    def compute_blind(self, asymm_obs=False, w_contact=True):  # The value net may get the priviliged info
        if asymm_obs:  
            # priveledged information for value net, which is not included in the policy net observation

            #dof pos
            self.states_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            # record dof vel
            self.states_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel

            hand_pose_start = 2*self.num_shadow_hand_dofs  # 44
            self.states_buf[:, hand_pose_start: hand_pose_start + 3] = self.hand_pos
            self.states_buf[:, hand_pose_start + 3: hand_pose_start + 7] = self.hand_rot

            obj_states_start = 2*self.num_shadow_hand_dofs + 7  # 51

            self.states_buf[:, obj_states_start:obj_states_start + 7] = self.object_pose


            #object linvel & angvel
            self.states_buf[:, obj_states_start + 7:obj_states_start + 10] = self.object_linvel
            self.states_buf[:, obj_states_start + 10:obj_states_start + 13] = self.vel_obs_scale * self.object_angvel

            # add part: goal pos & finger tips obs
            self.states_buf[:, obj_states_start + 13:obj_states_start + 16] = self.goal_pos

            ft_states_start = obj_states_start + 16  # 67

            num_ft_states = 13 * self.num_fingertips  # 65

            self.states_buf[:, ft_states_start:ft_states_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)


            contact_start = ft_states_start + num_ft_states # 132

            # ratio of active nodes in the contact patch, which can reflect the contact area
            self.states_buf[:, contact_start: contact_start + 1] = self.cps.unsqueeze(-1) / self.num_surface_points

            # contact vector: one-hot, 1 for contact and 0 for no contact
            self.states_buf[:, contact_start + 1: contact_start + self.contact_vector.shape[1] + 1] = self.contact_vector

            displacement_start = contact_start + self.contact_vector.shape[1] + 1 # 17 links, 150

            states_end = displacement_start + self.num_tac_features # 147 + 
            

            self.displacements[:, :, 0] /= (2.0 * self.feature_factor)
            self.displacements[:, :, 1] /= (2.0 * self.feature_factor)
            self.displacements[:, :, 2] /= (2.0 * self.feature_factor)

            # (B, n, 3) --> (B, n*3)
            self.states_buf[:, displacement_start: displacement_start + 3*self.num_tac_features] = self.displacements.reshape(self.num_envs, self.num_tac_features * 3)
            states_end = displacement_start + 3*self.num_tac_features # 150 + 3*num_tac_features 

            self.states_buf[:, states_end: states_end + self.num_actions] = self.actions # +24


        else: 
            # information for actor model, which we can observe in the implementation

            #dof pos
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits) # 22 dofs

            hand_pose_start = self.num_shadow_hand_dofs  # 22 
            self.obs_buf[:, hand_pose_start: hand_pose_start + 3] = self.hand_pos
            self.obs_buf[:, hand_pose_start + 3: hand_pose_start + 7] = self.hand_rot
            ft_obs_start = 1*self.num_shadow_hand_dofs + 7  # 29

            num_ft_obs = 13 * self.num_fingertips  # 65 for 5 fingertips
            
            self.obs_buf[:, ft_obs_start:ft_obs_start + num_ft_obs] = self.fingertip_state.reshape(self.num_envs, num_ft_obs)
            
            contact_start = ft_obs_start + num_ft_obs # 29+65=94
            
            # contact links vector: one-hot, 1 for contact and 0 for no contact
            self.obs_buf[:, contact_start: contact_start + self.contact_vector.shape[1]] = self.contact_vector

            displacement_obs = contact_start + self.contact_vector.shape[1] # 94+17=111

            self.displacements[:, :, 0] /= (2.0 * self.feature_factor)
            self.displacements[:, :, 1] /= (2.0 * self.feature_factor)
            self.displacements[:, :, 2] /= (2.0 * self.feature_factor)

            self.obs_buf[:, displacement_obs: displacement_obs + self.num_tac_features*3] = self.displacements.reshape(self.num_envs, self.num_tac_features*3)
            obs_end = displacement_obs + 3 * self.num_tac_features 

            self.obs_buf[:, obs_end: obs_end + self.num_actions] = self.actions # +24


    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        #self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        #goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12) 
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0
  
    def reset_idx(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)  #why plus 5? 3 for pos and 2 for rot

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object pos randomly
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone() # not necessary?
        highs = torch.tensor([0.03, 0.00])
        lows = torch.tensor([-0.03, -0.17])
        size = (self.hand_indices[env_ids].shape[0], 2)
        delta = ((highs - lows) * torch.rand(size) + lows).to(self.device)

        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.hand_start_states[env_ids, 0:2] + delta
        self.reset_position_noise = 0.01

        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = torch.zeros_like(self.object_init_state[env_ids, self.up_axis_idx]) - 0.02
        
        #random rotat in x and y axis
        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        if self.object_type == 'sodacan' or self.object_type == 'tank':
            new_object_rot = randomize_rotation(0.5*torch.ones_like(rand_floats[:, 3]), rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
            pass
        elif self.object_type == 'light':
            new_object_rot = randomize_rotation(0.2*rand_floats[:, 3], 0.2*rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        elif self.object_type == 'food':
            new_object_rot = randomize_rotation(0.2*rand_floats[:, 3], 0.2*rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
            pass

        #reset orientation and v a 
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])


        # get all object indices  att: NOT self.object_indices
        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand  random set?
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5:5+self.num_shadow_hand_dofs] + 1)

        #mini change of (default) dof pos
        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]
        #reset the target pos for dofs
        # prev and cur targets are for kinematics
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        #reset the dof states
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.root_state_tensor[self.hand_indices[env_ids], 0:7] = self.init_root_tensor[self.hand_indices[env_ids], 0:7]
        if self.object_type in self.start_heights.keys():
            self.root_state_tensor[self.hand_indices[env_ids], self.up_axis_idx] = self.start_heights[self.object_type]
        else:
            pass
        self.root_state_tensor[self.hand_indices[env_ids], 7:13] = torch.zeros_like(self.init_root_tensor[self.hand_indices[env_ids], 7:13])

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor), 
                                                     gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # add 4 log
        extras = dict()
        for key in self._episode_sums.keys():
            temp_sum = self._episode_sums[key][env_ids]/self.progress_buf[env_ids]
            episodic_sum_avg = torch.mean(temp_sum)
            extras["Detailed_Reward/" + key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    #
    def pre_physics_step(self, actions):
        #Apply the actions to the environment (eg by setting torques, position targets).


        '''actions1=torch.zeros_like(actions)
        actions1[:, 2] = -0.5
        actions = actions1'''

        #to judge whether there are envs in the buffer to be reset?
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset_idx()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)


        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        # num_actions: equal to the num of dofs?
        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            #dt: time gap
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            #make sure the dof state in the [lower bound, upper bound]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            '''
            @torch.jit.script
            def scale(x, lower, upper):
                return (0.5 * (x + 1.0) * (upper - lower) + lower)
            '''
            # so in this case, the action is used to set the state of each dof
            # 20 dofs as acurator in ShadowHand
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 6:],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,  #so act_moving_ave is a random weight? a weighted sum
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            
            # apply force and torque to the hand root
            self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000 # actor: hand
            self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
                                                    gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        ## TODO: modify this part, not only dof pos, the vel / angvel of hand need to be set as well
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0: #forceDecay: 0.99  force_decay_interval: 0.08, dt: 0.0167
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            #init: self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        # after action, get the reward and observation
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()  # get computed obs_buf 
        
        self.compute_reward(self.actions)
        
        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])


    def add_markers(self, contact_points, idx, color):
        '''
        Visualize the contact points. The radio of each marker is 0.0005
        Input:
            contact_points: list(num_cps, 3) -> all contact points 
            idx: the index of current env
        '''
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        sphere_geom = gymutil.WireframeSphereGeometry(0.001, 12, 12, sphere_pose, color=color)
        for point in contact_points:
            cp_pose = gymapi.Transform(p=gymapi.Vec3(point[0], point[1], point[2]))
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[idx], cp_pose)

    def remove_markers(self):
        self.gym.clear_lines(self.viewer)

    def draw_mesh(self, vertices, faces, idx, color):
        '''
        get lines to draw according to the mesh data
        vertices: list (num_vertices, 3)
        faces: list (num_faces, 3)
        return:
        np.array(num_lines, 6), each item in the array is [a.x, a.y, a.z, b.x, b.y, b.z] for line(a, b)
        '''
        lines = []
        for face in faces:
            #trimesh
            lines.append(vertices[face[0]]+vertices[face[1]])
            lines.append(vertices[face[0]]+vertices[face[2]])
            lines.append(vertices[face[1]]+vertices[face[2]])
        #lines = np.array(lines)
        for line in lines:
            self.gym.add_lines(self.viewer, self.envs[idx], 1, line, color)


    '''
    TODO: use modified hand model and object model in DexGraspNet to get the contact info
    Get the position, rotation and dof states of hand, to build up a 'vitual' model (i.e., hand_model)

    we use the hm to execute the rl?

    we have to modify the object model, for simple objects, we can see the object as one rigid body
    '''
    def setup_model(self):
        '''
        use the state of hands to setup a 'vitual' hand model
        get pos, rot and dof states of the hand, to set the paras of the hand model

        use hand and object model to calculate the contact info
        '''

        joint_names4hm = [
            'robot0:FFJ3',
            'robot0:FFJ2',
            'robot0:FFJ1',
            'robot0:FFJ0',
            'robot0:MFJ3',
            'robot0:MFJ2',
            'robot0:MFJ1',
            'robot0:MFJ0',
            'robot0:RFJ3',
            'robot0:RFJ2',
            'robot0:RFJ1',
            'robot0:RFJ0',
            'robot0:LFJ4',
            'robot0:LFJ3',
            'robot0:LFJ2',
            'robot0:LFJ1',
            'robot0:LFJ0',
            'robot0:THJ4',
            'robot0:THJ3',
            'robot0:THJ2',
            'robot0:THJ1',
            'robot0:THJ0'
        ]
        
        ###############################################################
        ###  There may be a difference between roots of shadow hand and 
        ###  shadow hand wrist free, we try to 
        ###
        ###  TODO： find out the root(or the best that can represent the root of it)
        ###         of shadow hand wrist free
        ###         use wrist as new root can approximate it, but is not accurate
        #############################################################

        hand_trans = self.root_state_tensor[self.hand_indices, 0:3]
        hand_rots = self.root_state_tensor[self.hand_indices, 3:7]
        hand_states = self.root_state_tensor[self.hand_indices, 0:13]
        
        num_links = self.etac.num_links
        ind2link = self.etac.ind2link
    
        
        joint_indexes = []
        for i,joint in enumerate(joint_names4hm):
            #get all 22 dof states
            joint_idx=self.gym.find_actor_dof_index(self.envs[0], self.shadow_hands[0]
                                                    , joint, gymapi.DOMAIN_ACTOR)
            joint_indexes.append(joint_idx)
            
        mid_dof_states = self.shadow_hand_dof_pos[:, joint_indexes].clone()
        

        object_states = torch.cat([self.object_pos, self.object_rot, self.object_linvel, self.object_angvel], dim=-1) # [B, 13]

        # get the states of each link
        link_states = torch.zeros((self.num_envs, num_links, 13), device=self.device)
        for i, link in enumerate(ind2link.keys()):
            link_states[:, i, :] = self.rigid_body_states[:, self.link_handles[i], :]     
        cps, contact_vector, displacements = self.etac.cal_tactile_info(hand_states, object_states, 
                                                                        mid_dof_states, link_states)

        
        return cps, contact_vector, displacements

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_grasp_reward(hand_pos, object_pos, num_surface_points, num_contacts, contact_links, fingertip_pos,
                         reset_buf, progress_buf, successes, current_successes, consecutive_successes,  
                         max_episode_length: float, target_pos, 
                         actions, out_dist: float, av_factor: float
    ):
    # PART 1, number of active nodes/larger contact area
    # Intuitively, larger contact area during grasp is more robust
    con_rew = 10 * num_contacts.squeeze()/num_surface_points


    # PART 2, distance between hand and object, our goal is to minimize this distance
    hand_dist = torch.norm(hand_pos - object_pos, p=2, dim=-1)


    # PART 3, encourage more hand link to reach the object.
    # similar to the contact graph of HOI
    num_links = torch.sum(contact_links, dim = -1).to(hand_dist.device)
    link_rew = 0.7 * num_links/contact_links.shape[1]

    # PART 4, lifting reward, inspired by UniDexGrasp (https://pku-epic.github.io/UniDexGrasp/)
    # encourages the robot hand to lift the object when the fingers are 
    # close enough to the object and the robot hand is considered to reach the target goal grasp pose.
    ft_dist = torch.zeros_like(hand_dist)
    for i in range(5):
        ft_dist += torch.norm(fingertip_pos[:, i] - object_pos, p=2, dim=-1)
    lift_flag = (hand_dist <= 0.12).int() + (ft_dist <= 0.25).int() + (num_contacts.squeeze()!=0).int()
    lift_rew = torch.zeros_like(hand_dist)
    # encourage to increase the force to let the hand go up when it is close enough to the obj
    lift_rew = torch.where(lift_flag >= 3, 1 * actions[:,2], lift_rew)

    # also, encourage the hand to lift
    goal_dist = target_pos[:, 2] - hand_pos[:, 2]
    goal_dist = torch.where(goal_dist < -0.3, -0.3, goal_dist)
    goal_hand_rew = torch.zeros_like(ft_dist)
    goal_hand_rew = torch.where(lift_flag >= 3, 0.5 * (1 - 2 * goal_dist), goal_hand_rew)

    # when reach the goal height, give a bouns
    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(lift_flag >= 3, torch.where(goal_dist <= 0.05, 0.5, bonus), bonus)
    
    # when the hand is not in contact with the obj, give a penalty
    nc_flag = (num_contacts.squeeze()==0).int()
    nc_pen = -0.5
    

    # PART 6(optional), a penalty if the hand did not lift the obj
    
    # when the hand is too far from the obj, which may be caused by the obj falling down or the hand missing the obj, we consider the task failed and 
    # give a large penalty.
    fail_flag = (hand_dist >= out_dist).int()
    penalty = -2.0



    # full reward
    reward = -1.0 * ft_dist - 0.8 * hand_dist + 1.5 * lift_rew + con_rew + link_rew + bonus + 3 * goal_hand_rew
    
    reward = torch.where(fail_flag == 1, reward + penalty, reward)
    reward = torch.where(nc_flag == 1, reward + nc_pen, reward)

    # add for log
    rewards = {
            "ft_dist": ft_dist.reshape(-1),
            "hand_dist": hand_dist.reshape(-1),
            "goal_hand_rew": goal_hand_rew.reshape(-1),
            "lift_rew": lift_rew.reshape(-1),
            "con_rew": con_rew.reshape(-1),
            "link_rew": link_rew.reshape(-1)
        }
    resets = reset_buf

    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    resets = torch.where(fail_flag == 1, torch.ones_like(resets), resets)
    goal_resets = resets

    success_flag = (goal_dist <= 0.05).int() + (num_contacts.squeeze()!=0).int() # the obj reaches the goal height and the hand is still in contact with it.

    # if the success grasping does not happen for a long time, we consider it as a failure and reset the env.
    # avoid repeated penalties
    early_reset_flag = (progress_buf >= 0.6 * max_episode_length).int() + (success_flag < 2).int() + (resets == 0).int()
    resets = torch.where(early_reset_flag==3, torch.ones_like(resets), resets)
    reward = torch.where(early_reset_flag==3, reward + penalty, reward)

    successes = torch.where(success_flag == 2, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    

    current_successes = torch.where(resets == 1, successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return rewards, reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


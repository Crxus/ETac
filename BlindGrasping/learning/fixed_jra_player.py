# player executing known trajectories
# 
# 
# 
import os
import shutil
import threading
import time
import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
import copy
import dill
from os.path import basename
from typing import Optional
from rl_games.common import vecenv
from rl_games.common import env_configurations
from rl_games.algos_torch import model_builder
from rl_games.common.player import BasePlayer
import hydra


class FixedPlayer(BasePlayer):
    def __init__(self, params):
        super().__init__(params)

        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)

        self.trajectories = None
        self.ends = None
        self.end_effector_pos = torch.tensor([])
        self.end_effector_rot = torch.tensor([])
        
    def set_trajectory(self, trajectories, ends):
        '''
        trajectories: [num_tra * dim_actions] a sequence of actions -> dof target positions for now
        ends: [num_episodes], indicates episode ends
        '''
        self.trajectories = trajectories
        self.ends = ends

    def run(self):
        '''
        we just run 1 env for this occaction?
        '''
        init_obs = self.env_reset(self.env)
        self.end_effector_pos.to(init_obs['obs'].device)
        self.end_effector_rot.to(init_obs['obs'].device)
        assert init_obs.shape[0]==1
        mark = 0
        for end in self.ends:
            _ = self.env_reset(self.env)
            for i in range(mark, end):
                action = self.trajectories[i]
                obs_dict, _, _, _ = self.env_step(self.env, action)
            # TBD: get the endeffector pos
            obs = obs_dict['obs'] #[1, dim_obs]
            ef_pos = obs[:, :3]
            ef_rot = obs[:, 3:6]
            self.end_effector_pos = torch.cat((self.end_effector_pos, ef_pos), dim=0)
            self.end_effector_rot = torch.cat((self.end_effector_rot, ef_rot), dim=0)
            mark = end

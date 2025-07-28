
import genesis as gs
import numpy as np
import torch
import yaml
import pandas as pd
from typing import Any
import types
from rsl_rl.env.vec_env import VecEnv
from tensordict import TensorDict

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Track_task(VecEnv):
    def __init__(self, genesis_env, env_config, task_config):
        # configs
        self.genesis_env = genesis_env
        self.env_config = env_config
        self.task_config = task_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # shapes
        self.num_envs = self.env_config.get("num_envs", 1)
        self.num_actions = task_config["num_actions"]
        self.num_commands = task_config["num_commands"]
        self.num_obs = task_config["num_obs"]

        # parameters
        self.max_episode_length = self.task_config.get("max_episode_length", 1500)
        self.reward_scales = task_config.get("reward_scales", {})
        self.obs_scales = task_config.get("obs_scales", {})
        self.command_cfg = self.env_config.get("command_cfg", {})
        self.step_dt = self.env_config.get("dt", 0.01)

        # buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.reward_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.command_buf = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.crash_condition = torch.ones((self.num_envs,), device=self.device, dtype=bool)
        self.cur_pos_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_pos_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.reward_functions = dict()
        self.episode_reward_sums = dict()
        self.extras = dict()  # extra information for logging

        self._register_reward_fun()

    def compute_reward(self):
        self.reward_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            reward = reward_func() * self.reward_scales[name]
            self.reward_buf += reward
            self.episode_reward_sums[name] += reward

    def _reward_target(self):
        target_reward = torch.sum(torch.square(self.last_pos_error), dim=1) - torch.sum(torch.square(self.cur_pos_error), dim=1)
        target_error_reward = -torch.norm(self.cur_pos_error, dim=1)
        return target_reward

    def _reward_smooth(self):
        smooth_reward = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_reward

    def _reward_yaw(self):
        yaw = self.genesis_env.drone.odom.body_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_reward = torch.exp(self.task_config["yaw_lambda"] * torch.abs(yaw))
        return yaw_reward

    def _reward_angular(self):
        angular_reward = torch.norm(self.genesis_env.drone.odom.body_ang_vel / 3.14159, dim=1)
        return angular_reward

    def _reward_crash(self):
        crash_reward = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_reward[self.crash_condition] = 1
        return crash_reward
    
    def _reward_lazy(self):
        lazy_reward = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        condition = self.genesis_env.drone.odom.world_pos[:, 2] < 0.1
        lazy_reward[condition] = self.episode_length_buf[condition] / self.max_episode_length
        
        return lazy_reward
        
    def _resample_commands(self, envs_idx):
        self.command_buf[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.command_buf[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.command_buf[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)

    def _register_reward_fun(self):
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.step_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_reward_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def _at_target(self):
        at_target = ((torch.norm(self.cur_pos_error, dim=1) < self.task_config["target_thr"]).nonzero(as_tuple=False).flatten())
        return at_target

    def step(self, action):
        self.actions = torch.clip(action, -self.task_config["clip_actions"], self.task_config["clip_actions"])
        exec_actions = self.actions
        
        if self.genesis_env.target is not None:
            self.genesis_env.target.set_pos(self.command_buf, zero_velocity=True, envs_idx=list(range(self.num_envs)))
        self.genesis_env.step(exec_actions)
        self.episode_length_buf += 1
        self.last_pos_error = self.command_buf - self.genesis_env.drone.odom.last_world_pos
        self.cur_pos_error = self.command_buf - self.genesis_env.drone.odom.world_pos
        self.crash_condition = (
            (torch.abs(self.genesis_env.drone.odom.body_euler[:, 1]) > self.task_config["termination_if_pitch_greater_than"])
            | (torch.abs(self.genesis_env.drone.odom.body_euler[:, 0]) > self.task_config["termination_if_roll_greater_than"])
            | (torch.abs(self.cur_pos_error[:, 0]) > self.task_config["termination_if_x_greater_than"])
            | (torch.abs(self.cur_pos_error[:, 1]) > self.task_config["termination_if_y_greater_than"])
            | (torch.abs(self.cur_pos_error[:, 2]) > self.task_config["termination_if_z_greater_than"])
            # | (self.genesis_env.drone.odom.world_pos[:, 2] < self.task_config["termination_if_close_to_ground"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition
        self.reset(self.reset_buf.nonzero(as_tuple=False).flatten())
        self.compute_reward()
        self.last_actions[:] = self.actions[:]
        self._resample_commands(self._at_target())
        
        self._update_obs()

        return self.get_observations(), self.reward_buf, self.reset_buf, self.extras


    def reset(self, env_idx=None):
        if env_idx is None:
            reset_range = torch.arange(self.num_envs, device=self.device)
        else:
            reset_range = env_idx

        self.genesis_env.reset(reset_range)
        self.last_actions[reset_range] = 0.0
        self.episode_length_buf[reset_range] = 0
        self.reset_buf[reset_range] = True

        self.extras["episode"] = {}
        for key in self.episode_reward_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_reward_sums[key][reset_range]).item() / self.task_config["episode_length_s"]
            )
            self.episode_reward_sums[key][reset_range] = 0.0
        self._resample_commands(reset_range)
        return self.get_observations()

    def get_observations(self):
        group_obs =  TensorDict({
            "state": self.obs_buf}, batch_size=self.num_envs
        )
        return group_obs


    def _update_obs(self):
        def check_nan(name, tensor):
            if torch.isnan(tensor).any():
                print(f"W[NaN DETECTED] {name} has NaN!")

        check_nan("cur_pos_error", self.cur_pos_error)
        check_nan("body_quat", self.genesis_env.drone.odom.body_quat)
        check_nan("world_linear_vel", self.genesis_env.drone.odom.world_linear_vel)
        check_nan("body_ang_vel", self.genesis_env.drone.odom.body_ang_vel)
        check_nan("last_actions", self.last_actions)

        self.obs_buf = torch.cat(
            [
                torch.clip(self.cur_pos_error * self.obs_scales["cur_pos_error"], -1, 1),
                self.genesis_env.drone.odom.body_quat,
                torch.clip(self.genesis_env.drone.odom.world_linear_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.genesis_env.drone.odom.body_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

    def get_privileged_observations(self):
        return None


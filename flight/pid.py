

import yaml
import torch
import argparse
import genesis as gs
import time
import threading
import numpy as np


class PIDcontroller:
    def __init__(self, num_envs, rc_command, odom, yaml_path, device = torch.device("cuda")):
        with open(yaml_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.rc_command = rc_command
        self.device = device
        self.num_envs = num_envs
        self.odom = odom
        self.use_RC = config.get("use_RC", False)

        # Shape: (n, 3)
        ang_cfg = config.get("ang", {})
        rat_cfg = config.get("rat", {})
        pos_cfg = config.get("pos", {})
        def get3(cfg, k1, k2, k3):
            return torch.tensor([
                cfg.get(k1, 0.0),
                cfg.get(k2, 0.0),
                cfg.get(k3, 0.0)
            ], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        # Angular controller (angle controller PID)
        self.kp_a = get3(ang_cfg, "kp_r", "kp_p", "kp_y")
        self.ki_a = get3(ang_cfg, "ki_r", "ki_p", "ki_y")
        self.kd_a = get3(ang_cfg, "kd_r", "kd_p", "kd_y")
        self.kf_a = get3(ang_cfg, "kf_r", "kf_p", "kf_y")
        self.P_term_a = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.I_term_a = torch.zeros_like(self.P_term_a)
        self.D_term_a = torch.zeros_like(self.P_term_a)
        self.F_term_a = torch.zeros_like(self.P_term_a)

        # Angular Rate controller (angular rate PID)
        self.kp_r = get3(rat_cfg, "kp_r", "kp_p", "kp_y")
        self.ki_r = get3(rat_cfg, "ki_r", "ki_p", "ki_y")
        self.kd_r = get3(rat_cfg, "kd_r", "kd_p", "kd_y")
        self.kf_r = get3(rat_cfg, "kf_r", "kf_p", "kf_y")
        self.P_term_r = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.I_term_r = torch.zeros_like(self.P_term_r)
        self.D_term_r = torch.zeros_like(self.P_term_r)
        self.F_term_r = torch.zeros_like(self.P_term_r)

        # Position controller (xyz and throttle)
        self.kp_p = get3(pos_cfg, "kp_x", "kp_y", "kp_t")
        self.ki_p = get3(pos_cfg, "ki_x", "ki_y", "ki_t")
        self.kd_p = get3(pos_cfg, "kd_x", "kd_y", "kd_t")
        self.kf_p = get3(pos_cfg, "kf_x", "kf_y", "kf_t")
        self.P_term_p = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.I_term_p = torch.zeros_like(self.P_term_p)
        self.D_term_p = torch.zeros_like(self.P_term_p)
        self.F_term_p = torch.zeros_like(self.P_term_p)                
        
        self.pid_freq = config.get("pid_exec_freq", 60)     # the same as gyro_update_freq
        self.base_rpm = config.get("base_rpm", 10000)
        self.dT = 1 / self.pid_freq
        self.tpa_factor = 1
        self.tpa_rate = 0

        self.last_body_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.angle_rate_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.angle_error = torch.zeros_like(self.angle_rate_error)

        self.body_set_point = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.pid_output = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone = None

    def set_drone(self, drone):
        self.drone = drone


    def mixer(self, action=torch.tensor(0.0)) -> torch.Tensor:
        throttle = torch.clamp(self.rc_command["throttle"] + action[:, -1]*0.4, 0, 1) * self.base_rpm * 3
        self.pid_output[:] = torch.clip(self.pid_output[:], -self.base_rpm * 3, self.base_rpm * 3)
        motor_outputs = torch.stack([
            throttle - self.pid_output[:, 0] - self.pid_output[:, 1] - self.pid_output[:, 2],  # M1
            throttle - self.pid_output[:, 0] + self.pid_output[:, 1] + self.pid_output[:, 2],  # M2
            throttle + self.pid_output[:, 0] + self.pid_output[:, 1] - self.pid_output[:, 2],  # M3
            throttle + self.pid_output[:, 0] - self.pid_output[:, 1] + self.pid_output[:, 2],  # M4
        ], dim = 1)

        return torch.clamp(motor_outputs, min=1, max=self.base_rpm * 3)  # size: tensor(num_envs, 4)

    def pid_update_TpaFactor(self):
        if (self.rc_command["throttle"] > 0.35):       # 0.35 is the tpa_breakpoint, the same as Betaflight, 
            if (self.rc_command["throttle"] < 1.0): 
                self.tpa_rate *= (self.rc_command["throttle"] - 0.35) / (1.0 - 0.35);            
            else:
                self.tpa_rate = 0.0
            self.tpa_factor = 1.0 - self.tpa_rate

    def step(self, action=torch.tensor(0.0)):
        self.odom.odom_update()
        # if(self.rc_command["ARM"] == 0):
        #     self.drone.set_propellels_rpm(torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float))
        #     return
        self.pid_update_TpaFactor()
        self.angle_controller(action)
        # if self.rc_command["ANGLE"] == 0:         # angle mode
        #     self.angle_controller(action)
        # elif self.rc_command["ANGLE"] == 1:       # angle rate mode
        #     self.rate_controller(action)
        # else:                                     # undifined
        #     print("undifined mode, do nothing!!")
        #     return
        self.drone.set_propellels_rpm(self.mixer(action))

    def rate_controller(self, action=torch.tensor(0.0)): 
        """
        Anglular rate controller, sequence is (roll, pitch, yaw), use previous-D-term PID controller

        :param: action: torch.Size([num_envs, 4]), like [[roll, pitch, yaw, thrust]] if num_envs = 1
        """
        if action.ndim == 0:
            self.body_set_point[:] = torch.tensor(list(self.rc_command.values())[:3]).repeat(self.num_envs, 1)
        else:
            self.body_set_point[:] = action[:, :3] + torch.tensor(list(self.rc_command.values())[:3]).repeat(self.num_envs, 1)      # index 1:roll, 2:pitch, 3:yaw, 4:throttle
        
        cur_angle_rate_error = self.body_set_point * 10 - self.odom.body_ang_vel
        self.P_term_r[:] = (cur_angle_rate_error * self.kp_r) * self.tpa_factor
        self.I_term_r[:] = self.I_term_r + cur_angle_rate_error * self.ki_r
        self.D_term_r[:] = (self.last_body_ang_vel - self.odom.body_ang_vel) * self.kd_r * self.tpa_factor    
        self.last_body_ang_vel[:] = self.odom.body_ang_vel
        self.pid_output[:] = (self.P_term_r + self.I_term_r + self.D_term_r)

    def angle_controller(self, action=torch.tensor(0.0)):  
        """
        Angle controller, sequence is (roll, pitch, yaw), use previous-D-term PID controller

        :param: action: torch.Size([num_envs, 4]), like [[roll, pitch, yaw, thrust]] if num_envs = 1
        """
        action = torch.as_tensor(action, dtype=gs.tc_float)
        if action.ndim == 0:
            self.body_set_point[:] = -self.odom.body_euler 
        else:
            self.body_set_point[:] = -self.odom.body_euler + action[:, :3]
        self.body_set_point[:] += torch.tensor(list(self.rc_command.values())[:3]).repeat(self.num_envs, 1)      # index 1:roll, 2:pitch, 3:yaw, 4:throttle
        cur_angle_rate_error = (self.body_set_point * 10 - self.odom.body_ang_vel)

        self.P_term_a[:] = (cur_angle_rate_error * self.kp_a) * self.tpa_factor
        self.I_term_a[:] = self.I_term_a + cur_angle_rate_error * self.ki_a
        self.D_term_a[:] = (self.last_body_ang_vel - self.odom.body_ang_vel) * self.kd_a * self.tpa_factor    
        # TODO feedforward term 
        self.last_body_ang_vel[:] = self.odom.body_ang_vel
        self.pid_output[:] = (self.P_term_a + self.I_term_a + self.D_term_a)


    # def position_controller(self, action):
    #     action = torch.as_tensor(action, dtype=gs.tc_float)
    #     self.body_set_point[:] = action
    #     cur_pos_error = (self.body_set_point * 10 - self.odom.body_ang_vel)

    #     self.P_term_p[:] = (cur_pos_error * self.kp_p)
    #     self.I_term_p[:] = self.I_term_p + cur_pos_error * self.ki_p
    #     self.D_term_p[:] = (self.last_body_ang_vel - self.odom.body_ang_vel) * self.kd_p  

    #     sum = self.P_term_p + self.I_term_p + self.D_term_p 

    #     self.angle_controller(self, sum)

    def reset(self, env_idx=None):
        if env_idx is None:
            reset_range = torch.arange(self.num_envs, device=self.device)
        else:
            reset_range = env_idx
        # Reset the PID terms (P, I, D, F)
        self.P_term_a.index_fill_(0, reset_range, 0.0)
        self.I_term_a.index_fill_(0, reset_range, 0.0)
        self.D_term_a.index_fill_(0, reset_range, 0.0)
        self.F_term_a.index_fill_(0, reset_range, 0.0)

        self.P_term_r.index_fill_(0, reset_range, 0.0)
        self.I_term_r.index_fill_(0, reset_range, 0.0)
        self.D_term_r.index_fill_(0, reset_range, 0.0)
        self.F_term_r.index_fill_(0, reset_range, 0.0)

        self.P_term_p.index_fill_(0, reset_range, 0.0)
        self.I_term_p.index_fill_(0, reset_range, 0.0)
        self.D_term_p.index_fill_(0, reset_range, 0.0)
        self.F_term_p.index_fill_(0, reset_range, 0.0)

        # Reset the angle, position, and velocity errors
        self.angle_rate_error.index_fill_(0, reset_range, 0.0)
        self.angle_error.index_fill_(0, reset_range, 0.0)

        # Reset the body set points and pid output
        self.body_set_point.index_fill_(0, reset_range, 0.0)
        self.pid_output.index_fill_(0, reset_range, 0.0)

        # Reset the last angular velocity
        self.last_body_ang_vel.index_fill_(0, reset_range, 0.0)

        # Reset the TPA factor and rate
        self.tpa_factor = 1
        self.tpa_rate = 0
        # Reset the RC command values if necessary
        if self.rc_command:
            self.rc_command["throttle"] = 0
            self.rc_command["ANGLE"] = 0
            self.rc_command["ARM"] = 0

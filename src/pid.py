

import yaml
import torch
import argparse
import genesis as gs
import time
import threading


class PIDcontroller:
    def __init__(self, env_num, rc_command, imu_sim, yaml_path, device = torch.device("cuda")):
        with open(yaml_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.rc_command = rc_command
        self.device = device
        self.env_num = env_num

        self.imu = imu_sim
        # Shape: (n, 3)
        self.kp = torch.tensor([config.get("kp_r", 0.0), config.get("kp_p", 0.0), config.get("kp_y", 0.0)], 
                                device=self.device).unsqueeze(0).repeat(self.env_num, 1)  
        self.ki = torch.tensor([config.get("ki_r", 0.0), config.get("ki_p", 0.0), config.get("ki_y", 0.0)], 
                                device=self.device).unsqueeze(0).repeat(self.env_num, 1)
        self.kd = torch.tensor([config.get("kd_r", 0.0), config.get("kd_p", 0.0), config.get("kd_y", 0.0)], 
                                device=self.device).unsqueeze(0).repeat(self.env_num, 1)
        self.kf = torch.tensor([config.get("kf_r", 0.0), config.get("kf_p", 0.0), config.get("kf_y", 0.0)], 
                                device=self.device).unsqueeze(0).repeat(self.env_num, 1)

        # self.entity_inertia = torch.tensor(config['inertia_matrix'], dtype=gs.tc_float)
        
        self.pid_freq = config.get("pid_exec_freq", 60)     # the same as gyro_update_freq
        self.base_rpm = config.get("base_rpm", 10000)
        self.dT = 1 / self.pid_freq
        self.tpa_factor = 1
        self.tpa_rate = 0

        
        self.last_body_ang_vel = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        # self.last_body_ang = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        # self.last_body_linear_vel = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        # self.last_body_acc = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)

        self.angle_rate_error = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.angle_error = torch.zeros_like(self.angle_rate_error)

        self.P_term = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.I_term = torch.zeros_like(self.P_term)
        self.D_term = torch.zeros_like(self.P_term)
        self.F_term = torch.zeros_like(self.P_term)

        self.body_set_point = torch.zeros((self.env_num, 4), device=self.device, dtype=gs.tc_float)
        self.pid_output = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.drone = None

    def set_entity(self, entity):
        self.drone = entity

    def mixer(self) -> torch.Tensor:
        throttle = (self.rc_command["throttle"] - 0.5) * 1000 

        # if (throttle < 0.9 * self.base_rpm):
        #     return torch.zeros((self.env_num, 4), device=self.device, dtype=gs.tc_float)
        

        motor_outputs = torch.stack([
            -self.pid_output[:, 0] - self.pid_output[:, 1] - self.pid_output[:, 2],  # M1
            -self.pid_output[:, 0] + self.pid_output[:, 1] + self.pid_output[:, 2],  # M2
             self.pid_output[:, 0] + self.pid_output[:, 1] - self.pid_output[:, 2],  # M3
             self.pid_output[:, 0] - self.pid_output[:, 1] + self.pid_output[:, 2],  # M4
        ], dim = 1)

        return motor_outputs + throttle + self.base_rpm


    def pid_update_TpaFactor(self):
        if (self.rc_command["throttle"] > 0.35):       # 0.35 is the tpa_breakpoint, the same as Betaflight, 
            if (self.rc_command["throttle"] < 1.0): 
                self.tpa_rate *= (self.rc_command["throttle"] - 0.35) / (1.0 - 0.35);            
            else:
                self.tpa_rate = 0.0
            
            self.tpa_factor = 1.0 - self.tpa_rate

    def controller_step(self):
        self.imu.imu_update()
        self.pid_update_TpaFactor()
        if self.rc_command["ANGLE"] == 0:         # angle mode
            self.angle_controller()
        elif self.rc_command["ANGLE"] == 1:       # angle rate mode
            self.angle_rate_controller()
        else:                                     # undifined
            print("undifined mode, do nothing!!")
            return
        # print([M1, M2, M3, M4])
        self.drone.set_propellels_rpm(self.mixer())


    def angle_rate_controller(self):  # use previous-D-term PID controller
        self.body_set_point[:] = torch.tensor(list(self.rc_command.values())[:4]).repeat(self.env_num, 1)      # index 1:roll, 2:pitch, 3:yaw, 4:throttle
        cur_angle_rate_error = self.body_set_point[:,0:3] - self.imu.body_ang_vel
        self.P_term[:] = (cur_angle_rate_error * self.kp) * self.tpa_factor
        self.I_term[:] = self.I_term + cur_angle_rate_error * self.ki
        self.D_term[:] = (self.last_body_ang_vel - self.imu.body_ang_vel) * self.kd * self.tpa_factor    
        # TODO feedforward term 
        self.last_body_ang_vel[:] = self.imu.body_ang_vel
        self.pid_output[:] = (self.P_term + self.I_term + self.D_term) * 100
        # print(self.pid_output)

    def angle_controller(self):  
        print("do nothing") 










    # def update_tpaFactor(self):

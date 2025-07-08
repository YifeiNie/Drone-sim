

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
        ang_cfg = config.get("ang", {})
        rat_cfg = config.get("rat", {})
        pos_cfg = config.get("pos", {})
        def get3(cfg, k1, k2, k3):
            return torch.tensor([
                cfg.get(k1, 0.0),
                cfg.get(k2, 0.0),
                cfg.get(k3, 0.0)
            ], device=self.device).unsqueeze(0).repeat(self.env_num, 1)
        
        # Angular controller (angle controller PID)
        self.kp_a = get3(ang_cfg, "kp_r", "kp_p", "kp_y")
        self.ki_a = get3(ang_cfg, "ki_r", "ki_p", "ki_y")
        self.kd_a = get3(ang_cfg, "kd_r", "kd_p", "kd_y")
        self.kf_a = get3(ang_cfg, "kf_r", "kf_p", "kf_y")
        self.P_term_a = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.I_term_a = torch.zeros_like(self.P_term_a)
        self.D_term_a = torch.zeros_like(self.P_term_a)
        self.F_term_a = torch.zeros_like(self.P_term_a)


        # Angular Rate controller (angular rate PID)
        self.kp_r = get3(rat_cfg, "kp_r", "kp_p", "kp_y")
        self.ki_r = get3(rat_cfg, "ki_r", "ki_p", "ki_y")
        self.kd_r = get3(rat_cfg, "kd_r", "kd_p", "kd_y")
        self.kf_r = get3(rat_cfg, "kf_r", "kf_p", "kf_y")
        self.P_term_r = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.I_term_r = torch.zeros_like(self.P_term_r)
        self.D_term_r = torch.zeros_like(self.P_term_r)
        self.F_term_r = torch.zeros_like(self.P_term_r)

        # Position controller (xyz and throttle)
        self.kp_p = get3(pos_cfg, "kp_x", "kp_y", "kp_t")
        self.ki_p = get3(pos_cfg, "ki_x", "ki_y", "ki_t")
        self.kd_p = get3(pos_cfg, "kd_x", "kd_y", "kd_t")
        self.kf_p = get3(pos_cfg, "kf_x", "kf_y", "kf_t")
        self.P_term_p = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.I_term_p = torch.zeros_like(self.P_term_p)
        self.D_term_p = torch.zeros_like(self.P_term_p)
        self.F_term_p = torch.zeros_like(self.P_term_p)                

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

        self.body_set_point = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.pid_output = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.drone = None

    def set_drone(self, drone):
        self.drone = drone


    def mixer(self) -> torch.Tensor:
        throttle = (self.rc_command["throttle"]) * self.base_rpm * 3

        # if (throttle < 0.9 * self.base_rpm):
        #     return torch.zeros((self.env_num, 4), device=self.device, dtype=gs.tc_float)
        
        motor_outputs = torch.stack([
            throttle - self.pid_output[:, 0] - self.pid_output[:, 1] - self.pid_output[:, 2],  # M1
            throttle - self.pid_output[:, 0] + self.pid_output[:, 1] + self.pid_output[:, 2],  # M2
            throttle + self.pid_output[:, 0] + self.pid_output[:, 1] - self.pid_output[:, 2],  # M3
            throttle + self.pid_output[:, 0] - self.pid_output[:, 1] + self.pid_output[:, 2],  # M4
        ], dim = 1)

        return torch.clamp(motor_outputs, min=1, max=self.base_rpm * 5)  # size: tensor(env_num, 4)

    def pid_update_TpaFactor(self):
        if (self.rc_command["throttle"] > 0.35):       # 0.35 is the tpa_breakpoint, the same as Betaflight, 
            if (self.rc_command["throttle"] < 1.0): 
                self.tpa_rate *= (self.rc_command["throttle"] - 0.35) / (1.0 - 0.35);            
            else:
                self.tpa_rate = 0.0
            
            self.tpa_factor = 1.0 - self.tpa_rate


    def controller_step(self):
        self.imu.imu_update()
        # if(self.rc_command["ARM"] == 0):
        #     self.drone.set_propellels_rpm(torch.zeros((self.env_num, 4), device=self.device, dtype=gs.tc_float))
        #     return
        self.pid_update_TpaFactor()
        if self.rc_command["ANGLE"] == 0:         # angle mode
            self.angle_controller()
        elif self.rc_command["ANGLE"] == 1:       # angle rate mode
            self.rate_controller()
        else:                                     # undifined
            print("undifined mode, do nothing!!")
            return
        # print([M1, M2, M3, M4])
        self.drone.set_propellels_rpm(self.mixer())


    def rate_controller(self, command=0):  # use previous-D-term PID controller
        command = torch.as_tensor(command, dtype=gs.tc_float)
        self.body_set_point[:] = command + torch.tensor(list(self.rc_command.values())[:3]).repeat(self.env_num, 1)      # index 1:roll, 2:pitch, 3:yaw, 4:throttle
        cur_angle_rate_error = self.body_set_point * 10 - self.imu.body_ang_vel

        self.P_term_r[:] = (cur_angle_rate_error * self.kp_r) * self.tpa_factor
        self.I_term_r[:] = self.I_term_r + cur_angle_rate_error * self.ki_r
        self.D_term_r[:] = (self.last_body_ang_vel - self.imu.body_ang_vel) * self.kd_r * self.tpa_factor    
        # TODO feedforward term 
        self.last_body_ang_vel[:] = self.imu.body_ang_vel
        self.pid_output[:] = (self.P_term_r + self.I_term_r + self.D_term_r)


    def angle_controller(self, command=0):  
        command = torch.as_tensor(command, dtype=gs.tc_float)
        self.body_set_point[:] = -self.imu.body_euler + command
        self.body_set_point[:] += torch.tensor(list(self.rc_command.values())[:3]).repeat(self.env_num, 1)      # index 1:roll, 2:pitch, 3:yaw, 4:throttle
        cur_angle_rate_error = (self.body_set_point * 10 - self.imu.body_ang_vel)

        self.P_term_a[:] = (cur_angle_rate_error * self.kp_a) * self.tpa_factor
        self.I_term_a[:] = self.I_term_a + cur_angle_rate_error * self.ki_a
        self.D_term_a[:] = (self.last_body_ang_vel - self.imu.body_ang_vel) * self.kd_a * self.tpa_factor    
        # TODO feedforward term 
        self.last_body_ang_vel[:] = self.imu.body_ang_vel
        self.pid_output[:] = (self.P_term_a + self.I_term_a + self.D_term_a)


    def position_controller(self, command):
        command = torch.as_tensor(command, dtype=gs.tc_float)
        self.body_set_point[:] = command
        cur_pos_error = (self.body_set_point * 10 - self.imu.body_ang_vel)

        self.P_term_p[:] = (cur_pos_error * self.kp_p)
        self.I_term_p[:] = self.I_term_p + cur_pos_error * self.ki_p
        self.D_term_p[:] = (self.last_body_ang_vel - self.imu.body_ang_vel) * self.kd_p  

        sum = self.P_term_p + self.I_term_p + self.D_term_p 

        self.angle_controller(self, sum)






    # def update_tpaFactor(self):

import threading
import yaml
import time
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

# NOTE :IMU is not an odomtry, only has anglear_velocity and linear_accerleration
class Odom:
    def __init__(self, num_envs, device = torch.device("cuda")):
        self.device = device
        self.drone = None
        self.num_envs = num_envs

        # body data
        self.body_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.body_linear_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.body_linear_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)        
        self.body_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.body_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.body_quat_inv = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.last_body_linear_vel = torch.zeros_like(self.body_linear_vel)    # used to cal acc

        # global data
        self.world_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.world_linear_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.world_linear_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.world_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.world_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.last_world_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_world_linear_vel = torch.zeros_like(self.body_linear_vel)
        self.last_time = time.perf_counter()

        # for Betaflight, abort!
        # gyro_sample_freq = 8kHz, gyro_filtered_freq = gyro_sample_freq / pid_process_denom = 4kHz
        # acc_sample_freq = 1kHz, acc_target_freq = acc_sample_freq / pid_process_denom = = 500Hz
        # attitude estimate freq = 100Hz / imu_process_denom
        # pid_process_denom can found by "get pid_process_denom" in cli in Betaflight configrator
        # imu_process_denom can found by "get imu_process_denom" in cli in Betaflight configrator
        
        # self.gyro_update_freq = config.get("gyro_update_freq", 8000)   # default are the same as Betaflight, aborted!
        # self.acc_update_freq = config.get("acc_update_freq", 1000)  
        # self.att_update_freq = config.get("att_update_freq", 100)
       
        # self.gyro_update_dT = 1 / self.gyro_update_freq
        # self.acc_update_dT = 1 / self.acc_update_freq
        # self.att_update_dT = 1 / self.att_update_freq

    def set_drone(self, drone):
        self.drone = drone

    def cal_cur_quat(self):
        self.body_quat[:] = self.drone.get_quat()
        self.body_quat_inv[:] = inv_quat(self.body_quat)

    def gyro_update(self):
        self.cal_cur_quat()    # since gyro_update has the highest freq
        cur_ang_vel = self.drone.get_ang()      # (roll, pitch, yaw)
        self.body_ang_vel[:] = cur_ang_vel
        self.world_ang_vel[:] = transform_by_quat(cur_ang_vel, self.body_quat_inv)

    def acc_update(self, dT):
        self.last_body_linear_vel[:] = self.body_linear_vel
        body_linear_vel_temp = self.drone.get_vel()
        self.body_linear_vel[:] = body_linear_vel_temp
        self.world_linear_vel[:] = transform_by_quat(body_linear_vel_temp, self.body_quat_inv)

        self.body_linear_acc[:] = (self.body_linear_vel - self.last_body_linear_vel) / dT
        self.world_linear_acc[:] = transform_by_quat(self.body_linear_acc, self.body_quat_inv)

    def att_update(self):
        self.body_euler[:] = quat_to_xyz(self.body_quat, rpy=True)

    def pos_update(self):
        self.last_world_pos[:] = self.world_pos[:]
        self.world_pos[:] = self.drone.get_pos()

    def odom_update(self):
        self.gyro_update()
        self.acc_update(time.perf_counter() - self.last_time)
        self.att_update()
        self.pos_update()
        self.last_time = time.perf_counter()
        
    def reset(self, env_idx):
        if env_idx is None:
            reset_range = torch.arange(self.num_envs, device=self.device)
        else:
            reset_range = env_idx
        # Reset body data to zero
        self.body_euler[reset_range] = torch.zeros_like(self.body_euler)
        self.body_linear_vel[reset_range] = torch.zeros_like(self.body_linear_vel)
        self.body_linear_acc[reset_range] = torch.zeros_like(self.body_linear_acc)
        self.body_ang_vel[reset_range] = torch.zeros_like(self.body_ang_vel)
        self.body_quat[reset_range] = torch.zeros_like(self.body_quat)
        self.body_quat_inv[reset_range] = torch.zeros_like(self.body_quat_inv)
        self.last_body_linear_vel[reset_range] = torch.zeros_like(self.last_body_linear_vel)

        # Reset global data to zero
        self.world_euler[reset_range] = torch.zeros_like(self.world_euler)
        self.world_linear_vel[reset_range] = torch.zeros_like(self.world_linear_vel)
        self.world_linear_acc[reset_range] = torch.zeros_like(self.world_linear_acc)
        self.world_pos[reset_range] = torch.zeros_like(self.world_pos)
        self.world_ang_vel[reset_range] = torch.zeros_like(self.world_ang_vel)

        self.last_world_pos[reset_range] = torch.zeros_like(self.last_world_pos)
        self.last_world_linear_vel[reset_range] = torch.zeros_like(self.last_world_linear_vel)

        # Reset the time
        self.last_time = time.perf_counter()



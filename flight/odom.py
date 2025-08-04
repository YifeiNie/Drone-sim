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
        self.has_none = torch.zeros((self.num_envs,), device=self.device, dtype=bool)

        # body data
        self.body_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.body_linear_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.body_linear_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)        
        self.body_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float)
        self.body_quat = identity_quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.body_quat_inv = identity_quat.unsqueeze(0).repeat(self.num_envs, 1)

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

    def set_drone(self, drone):
        self.drone = drone

    def cal_cur_quat(self):
        self.body_quat[:] = self.drone.get_quat()
        self.has_none[:] = torch.isnan(self.body_quat).any(dim=1)
        if (torch.any(self.has_none)):
            print("get_quat NaN env_idx:", torch.nonzero(self.has_none).squeeze())
            self.body_quat_inv[self.has_none] = inv_quat(self.body_quat[self.has_none])
        else:
            self.body_quat_inv = inv_quat(self.body_quat)

    def gyro_update(self):
        self.cal_cur_quat()    
        cur_ang_vel = self.drone.get_ang()      # (roll, pitch, yaw)
        self.body_ang_vel[:] = cur_ang_vel
        self.world_ang_vel[:] = transform_by_quat(cur_ang_vel, self.body_quat_inv)

    def acc_update(self, dT):
        self.last_body_linear_vel[:] = self.body_linear_vel
        world_linear_vel_temp = self.drone.get_vel()
        self.world_linear_vel[:] = world_linear_vel_temp
        self.body_linear_vel[:] = transform_by_quat(world_linear_vel_temp, self.body_quat_inv)

        self.body_linear_acc[:] = (self.body_linear_vel - self.last_body_linear_vel) / dT
        self.world_linear_acc[:] = transform_by_quat(self.body_linear_acc, self.body_quat)

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
        self.body_euler.index_fill_(0, reset_range, 0.0)
        self.body_linear_vel.index_fill_(0, reset_range, 0.0)
        self.body_linear_acc.index_fill_(0, reset_range, 0.0)
        self.body_ang_vel.index_fill_(0, reset_range, 0.0)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float)
        self.body_quat[reset_range] = identity_quat.unsqueeze(0).repeat(len(reset_range), 1)
        self.body_quat_inv[reset_range] = identity_quat.unsqueeze(0).repeat(len(reset_range), 1)

        self.last_body_linear_vel.index_fill_(0, reset_range, 0.0)

        # Reset global data to zero
        self.world_euler.index_fill_(0, reset_range, 0.0)
        self.world_linear_vel.index_fill_(0, reset_range, 0.0)
        self.world_linear_acc.index_fill_(0, reset_range, 0.0)
        self.world_pos.index_fill_(0, reset_range, 0.0)
        self.world_ang_vel.index_fill_(0, reset_range, 0.0)

        self.last_world_pos.index_fill_(0, reset_range, 0.0)
        self.last_world_linear_vel.index_fill_(0, reset_range, 0.0)

        # Reset the time
        self.last_time = time.perf_counter()



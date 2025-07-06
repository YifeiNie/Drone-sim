import threading
import yaml
import time
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

# NOTE :IMU is not an odomtry, only has anglear_velocity and linear_accerleration
class IMU_sim:
    def __init__(self, env_num, yaml_path, device = torch.device("cuda")):
        with open(yaml_path, "r") as file:
            config = yaml.load(file, Loader = yaml.FullLoader)
        self.device = device
        self.drone = None
        self.env_num = env_num
        
        # self.last_body_pos = torch.zeros_like(self.body_pos)
        # self.body_pos = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)

        self.body_euler = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.body_linear_vel = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.body_linear_acc = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)        
        self.body_ang_vel = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.last_body_linear_vel = torch.zeros_like(self.body_linear_vel)    # used to cal acc

        self.body_quat = torch.zeros((self.env_num, 4), device=self.device, dtype=gs.tc_float)
        self.body_quat_inv = torch.zeros((self.env_num, 4), device=self.device, dtype=gs.tc_float)

        self.world_euler = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.world_linear_vel = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.world_linear_acc = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.last_world_linear_vel = torch.zeros_like(self.body_linear_vel)
        self.world_ang_vel = torch.zeros((self.env_num, 3), device=self.device, dtype=gs.tc_float)
        self.last_time = time.perf_counter()


        # for Betaflight
        # gyro_sample_freq = 8kHz, gyro_filtered_freq = gyro_sample_freq / pid_process_denom = 4kHz
        # acc_sample_freq = 1kHz, acc_target_freq = acc_sample_freq / pid_process_denom = = 500Hz
        # attitude estimate freq = 100Hz / imu_process_denom
        # pid_process_denom can found by "get pid_process_denom" in cli in Betaflight configrator
        # imu_process_denom can found by "get imu_process_denom" in cli in Betaflight configrator
        
        self.gyro_update_freq = config.get("gyro_update_freq", 8000)   # default are the same as Betaflight, aborted!
        self.acc_update_freq = config.get("acc_update_freq", 1000)  
        self.att_update_freq = config.get("att_update_freq", 100)
       
        self.gyro_update_dT = 1 / self.gyro_update_freq
        self.acc_update_dT = 1 / self.acc_update_freq
        self.att_update_dT = 1 / self.att_update_freq

        self.tasks = {
            "gyro_update": {
                "interval": 1.0 / self.gyro_update_freq,  # 8000Hz
                "last_time": time.perf_counter(),
                "func": self.gyro_update
            },
            "acc_update": {
                "interval": 1.0 / self.acc_update_freq,   # 1000Hz
                "last_time": time.perf_counter(),
                "func": self.acc_update
            },
            "att_update": {
                "interval": 1.0 / self.att_update_freq,   # 100Hz
                "last_time": time.perf_counter(),
                "func": self.att_update
            }
        }

    def set_drone(self, drone):
        self.drone = drone

    def cal_cur_quat(self):
        self.body_quat[:] = self.drone.get_quat()
        self.body_quat_inv[:] = inv_quat(self.body_quat)

    def gyro_update(self):
        self.cal_cur_quat()    # since gyro_update has the highest freq
        cur_ang_vel = self.drone.get_ang()
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

    # a simple scheduler to sim Betaflight
    def imu_sim_loop(self):
        while True:
            now = time.perf_counter()
            next_wakeup = float("inf")

            for task_name, task in self.tasks.items():
                elapsed = now - task["last_time"]
                if elapsed >= task["interval"]:
                    task["func"]()
                    task["last_time"] = now
                else:
                    time_to_next = task["interval"] - elapsed
                    next_wakeup = min(next_wakeup, time_to_next)

            # sleep until the next nearest task
            if next_wakeup > 0:
                time.sleep(next_wakeup)

    # NOTE Abort! since Genesis cannot update scene so quick
    def start_imu_sim_thread(self):
        t = threading.Thread(target=self.imu_sim_loop, daemon=True)
        t.start()

    # update imu every step (defalt in 60Hz)
    def imu_update(self):
        self.gyro_update()
        self.acc_update(time.perf_counter() - self.last_time)
        self.att_update()
        self.last_time = time.perf_counter()
        
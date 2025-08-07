

import shutil
import os
import yaml
import torch
from flight.pid import PIDcontroller
from flight.odom import Odom
from flight.mavlink_sim import rc_command
from env.genesis_env import Genesis_env
from flight.mavlink_sim import start_mavlink_receive_thread
from algorithms.rl.tasks.track_task import Track_task
import time
from datetime import datetime
import genesis as gs
import warp as wp

def gs_rand_float(lower, upper, device="cuda"):
    shape = lower.shape  # scalar
    return (upper - lower) * torch.rand(size=shape, device=device) + lower



def main():
    # logging_level="warning"
    gs.init(logging_level="warning")
    num_envs = 1
    command_buf = torch.zeros((num_envs, 4), device="cuda", dtype=gs.tc_float)
    def update_commands(cur_pos, envs_idx=None):
        if envs_idx is None:
            idx = torch.arange(num_envs, "cuda")
        else:
            idx = envs_idx
        command_buf[idx, 0] = gs_rand_float(cur_pos[idx, 0]-0.2, cur_pos[idx, 0]+0.2)
        command_buf[idx, 1] = gs_rand_float(cur_pos[idx, 1]-0.2, cur_pos[idx, 1]+0.2)
        command_buf[idx, 2] = gs_rand_float(torch.clamp(cur_pos[idx, 2]-0.2, min=0.3, max=2.0), cur_pos[idx, 2]+0.2)

    
    def at_target(cur_pos):
        cur_pos_error = cur_pos - command_buf[:, :3]
        at_target = ((torch.norm(cur_pos_error, dim=1) < 0.2).nonzero(as_tuple=False).flatten())
        return at_target

    with open("config/demos/pos_ctrl_eval/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    with open("config/demos/pos_ctrl_eval/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)


    genesis_env = Genesis_env(
        env_config = env_config, 
        flight_config = flight_config,
    )

    device = "/dev/ttyUSB0"
    if not os.path.exists(device):
        print(f"[MAVLINK] Device {device} not found, skipping mavlink thread.")
    else :
        start_mavlink_receive_thread(device)

    while True:
        cur_pos = genesis_env.drone.odom.world_pos
        update_commands(cur_pos, at_target(cur_pos))
        genesis_env.target.set_pos(command_buf[:, :3], zero_velocity=True, envs_idx=list(range(num_envs)))
        genesis_env.step(command_buf)

if __name__ == "__main__" :
    wp.config.enable_backward_log = True
    main()


    
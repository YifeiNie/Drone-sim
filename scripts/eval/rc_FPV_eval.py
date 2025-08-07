

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
from rsl_rl.runners import OnPolicyRunner
import time
from datetime import datetime
import genesis as gs
import warp as wp

def main():

    # logging_level="warning"
    gs.init(logging_level="warning")
    
    with open("config/demos/rc_FPV_eval/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    with open("config/demos/rc_FPV_eval/flight.yaml", "r") as file:
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
        genesis_env.step()

if __name__ == "__main__" :
    wp.config.enable_backward_log = True
    main()


    
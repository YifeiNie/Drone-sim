

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

def main():

    # logging_level="warning"
    gs.init(logging_level="warning")


    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_dir = os.path.join(current_dir, f"logs/track_task_{timestamp}")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)


    with open("config/demos/sim_env/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    with open("config/demos/sim_env/flight.yaml", "r") as file:
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


    
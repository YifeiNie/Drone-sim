

import shutil
import os
import yaml
import torch
from flight.pid import PIDcontroller
from flight.odom import Odom
from flight.mavlink_sim import rc_command
from env.genesis_env import Genesis_env
from flight.mavlink_sim import start_mavlink_receive_thread
from learning.rl.tasks.track_task import Track_task
from rsl_rl.runners import OnPolicyRunner
import time
from datetime import datetime
import genesis as gs
import warp as wp

def main():
    # logging_level="warning"
    gs.init()


    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_dir = os.path.join(current_dir, f"logs/rsl_track/track_task_{timestamp}")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)


    with open("config/sim_env/env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/rl_task/track.yaml", "r") as file:
        rl_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/sim_env/flight.yaml", "r") as file:
        controller_config = yaml.load(file, Loader=yaml.FullLoader)

    task_config = rl_config["task"]
    train_config = rl_config["train"]


    genesis_env = Genesis_env(
        env_config = env_config, 
        controller_config = controller_config,
    )

    track_task = Track_task(
        genesis_env = genesis_env, 
        env_config = env_config, 
        task_config = task_config
    )

    runner = OnPolicyRunner(track_task, train_config, log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=500, init_at_random_ep_len=True)

if __name__ == "__main__" :
    wp.config.enable_backward_log = True
    main()


    




import yaml
import torch
from flight.pid import PIDcontroller
from flight.imu_sim import IMU_sim
from flight.mavlink_sim import rc_command
from env.genesis_env import Genesis_env
from flight.mavlink_sim import start_mavlink_receive_thread
import time
import genesis as gs
import warp as wp

def main():
    gs.init()
    with open("config/flight/env.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    genesis_env = Genesis_env(
        num_envs = config.get("num_envs", 1),
        yaml_path = "config/flight/env.yaml",
        drone = gs.morphs.Drone(file="urdf/cf2x.urdf", pos=(0.0, 0.0, 0.0)),
        device = torch.device("cuda")
    )

    start_mavlink_receive_thread()
    while True:
        genesis_env.sim_step()

if __name__ == "__main__" :
    wp.config.enable_backward_log = True
    main()


    
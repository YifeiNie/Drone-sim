



import yaml
import torch
from flight.pid import PIDcontroller
from flight.imu_sim import IMU_sim
from flight.mavlink_sim import rc_command
from env.test_env import Test_env
from flight.mavlink_sim import start_mavlink_receive_thread
import time
import genesis as gs
import warp as wp

def main():
    gs.init()
    with open("config/env.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    test_env = Test_env(
        env_num = config.get("env_num", 1),
        yaml_path = "config/env.yaml",
        drone = gs.morphs.Drone(file="urdf/cf2x.urdf", pos=(0.0, 0.0, 0.0)),
        device = torch.device("cuda")
    )

    start_mavlink_receive_thread()
    while True:
        test_env.sim_step()

if __name__ == "__main__" :
    wp.config.enable_backward_log = True
    main()


    
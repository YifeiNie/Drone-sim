



import yaml
import torch
from src.pid import PIDcontroller
from src.imu_sim import IMU_sim
from src.mavlink_sim import rc_command
from env.test_env import Test_env
from src.mavlink_sim import start_mavlink_receive_thread
import time
import genesis as gs

def main():
    gs.init()
    with open("config/env.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    imu = IMU_sim(
        env_num = config.get("num_env", 1),
        entity = None,
        yaml_path = "config/imu_sim_param.yaml",
        device = torch.device("cuda")
    )

    pid = PIDcontroller(
        env_num = config.get("num_env", 1), 
        rc_command = rc_command,
        imu_sim = imu, 
        yaml_path = "config/pid_param.yaml",
        device = torch.device("cuda")
    )

    test_env = Test_env(
        num_envs = config.get("num_env", 1),
        yaml_path = "config/env.yaml",
        controller = pid,
        entity = gs.morphs.Drone(file="urdf/cf2x.urdf", pos=(0.0, 0.0, 0.0)),
        device = torch.device("cuda")
    )

    imu.set_entity(test_env.get_entity())
    pid.set_entity(test_env.get_entity())
    start_mavlink_receive_thread()
    while True:
        test_env.sim_step()

if __name__ == "__main__" :
    main()


    
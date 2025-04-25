



import yaml
import torch
from src.pid import PIDcontroller
from src.imu_sim import IMU_sim
from src.mavlink_sim import rc_command
from env.test_env import Test_env
import genesis as gs

def main():
    gs.init()
    with open("config/env.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    test_env = Test_env(
        num_envs = config.get("num_env", 1),
        yaml_path = "config/env.yaml",
        device = torch.device("cuda")
    )

    imu = IMU_sim(
        env_num = config.get("num_env", 1),
        entity = drone_entity,
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



if __name__ == "__main__" :
    main()


    
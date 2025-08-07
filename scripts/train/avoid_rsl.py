

import shutil
import os
import yaml
from env.genesis_env import Genesis_env
from algorithms.rl.tasks.avoid_task import Avoid_task
from rsl_rl.runners import OnPolicyRunner
from datetime import datetime
import genesis as gs
import warp as wp

def main():
    # logging_level="warning"
    gs.init(logging_level="warning")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_dir = f"logs/avoid_rsl/avoid_{timestamp}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)


    with open("config/tasks/avoid_rsl/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/tasks/avoid_rsl/rl_env.yaml", "r") as file:
        rl_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/tasks/avoid_rsl/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)

    task_config = rl_config["task"]
    train_config = rl_config["train"]


    genesis_env = Genesis_env(
        env_config = env_config, 
        flight_config = flight_config,
    )

    avoid_task = Avoid_task(
        genesis_env = genesis_env, 
        env_config = env_config, 
        task_config = task_config,
        train_config = train_config,
    )
    genesis_env.step()
    runner = OnPolicyRunner(avoid_task, train_config, log_dir, device="cuda:0")
    avoid_task.networks["actor"] = runner.alg.policy.actor
    avoid_task.networks["critic"] = runner.alg.policy.critic
    runner.learn(num_learning_iterations=train_config["max_iterations"], init_at_random_ep_len=True)

if __name__ == "__main__" :
    wp.config.enable_backward_log = True
    main()


    
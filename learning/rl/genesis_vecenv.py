from stable_baselines3.common.vec_env import VecEnv
import numpy as np
import torch

class GenesisVecEnv(VecEnv):
    def __init__(self, genesis_scene, num_envs, observation_space, action_space):
        super().__init__(num_envs, observation_space, action_space)
        self.genesis_scene = genesis_scene
        self.num_envs = num_envs
        self.actions = torch.zeros((num_envs, 4))

    def reset(self):
        # 一次性 reset 所有 agent
        obs = self.genesis_scene.reset()  # 返回 shape = (num_envs, obs_dim)
        return obs

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        # 传入所有 agent 的动作，执行一次仿真
        # 都是np.stack
        obs, rewards, dones, infos = self.genesis_scene.step(torch.from_numpy(self.actions))

        # 把 dones 为 True 的 agent reset（Genesis 内部需要支持部分 reset）
        for i in range(self.num_envs):
            if dones[i]:
                obs[i] = self.genesis_scene.reset_agent(i)  # 局部 reset 某个 agent

        return obs, rewards, dones, infos

    def close(self):
        self.genesis_scene.close()

    def seed(self, seed=None):
        self.genesis_scene.seed(seed)

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.genesis_scene, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.genesis_scene, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self.genesis_scene, method_name)
        return [method(*method_args, **method_kwargs)] * self.num_envs

import genesis as gs
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import yaml
import pandas as pd
from typing import Any


class track_task(gym.Env):
    def __init__(
        self, 
        env = None, 
        env_config_path = None, 
        learning_config_path = None, 
        num_envs = 1,
        agent_num = 1,
        episode_len = 15, 
        sim_freq = 100, 
        ctrl_freq = 100
    ):
        
        super().__init__()
        self.env = env, 
        self.num_envs = num_envs,
        self.agent_num = agent_num,
        self.episode_len = episode_len, 
        self.sim_freq = sim_freq, 
        self.ctrl_freq = ctrl_freq
        self.np_random = None
        
        with open(env_config_path, "r") as file:
            self.env_config = yaml.load(file, Loader=yaml.FullLoader)

        with open(learning_config_path, "r") as file:
            self.learning_config = yaml.load(file, Loader=yaml.FullLoader)

        self.action_space = self._action_space()

        self.observation_space = self._observation_space()


    def _action_space(self) -> spaces.Box:
        agent_action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),    # (roll, pitch, yaw, thrust)
            high=np.array([1, 1, 1, 1], dtype=np.float32),
        )
        return agent_action_space

    def _observation_space(self) -> spaces.Box:
        angent_observation_space = spaces.Box(
            low=0, 
            high=255,
            shape=(self.learning_config.output_dim + self.learning_config.obs_dim, ),  # as a vector
            dtype=np.float32)    
        return angent_observation_space
    

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        aabb_list = self.env.get_aabb_list()


        observation = self._observation_space()


        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...


















    def ensure_min_distance(self, track_data, min_distance):
        """
        Ensure that consecutive points along the trajectory are separated by at least
        a specified spatial distance, while keeping their time stamps in order.

        :param track_data: Discrete trajectory samples, e.g.
                           [(t1, x1, y1, z1), (t2, x2, y2, z2), ...]
        :param min_distance: Minimum allowed distance between successive points (meters)
        :return: Filtered trajectory points satisfying the distance constraint
        """
        adjusted_data = [track_data[0]]
        for i in range(1, len(track_data)):
            prev_point = adjusted_data[-1]
            curr_point = track_data[i]
            distance = np.linalg.norm(curr_point[1:] - prev_point[1:])
            if distance >= min_distance:
                adjusted_data.append(curr_point)
        adjusted_data = np.array(adjusted_data)
        return (
            adjusted_data[:, 0],        # timestamp
            adjusted_data[:, 1:4],      # position
            adjusted_data[:, 4:7],      # speed 
            adjusted_data[:, 7:11],     # quaternion
            adjusted_data[:, 11:14],    # angular 
            np.sum(adjusted_data[:, 14:18], axis=1) / 0.85 * self.M,    # thrust
        )  

    def calculate_tangent_vectors(self):
        """
        calculate tangent

        :param track_points: position way points, like [(x1, y1, z1), (x2, y2, z2), ...]
        :return: tangent array
        """
        track_points = self.track_points
        tangent_vectors = np.zeros_like(self.track_points)
        tangent_vectors[0] = track_points[1] - track_points[0]
        tangent_vectors[1:-1] = track_points[2:] - track_points[:-2]
        tangent_vectors[-1] = track_points[-1] - track_points[-2]
        norms = np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
        tangent_unit_vectors = tangent_vectors / norms

        return tangent_unit_vectors
    

    def read_track_points(self, track_points_path):
        """
        read waypoints from csv

        :param track_path: csv file path
        :return: path array, like [(t1, x1, y1, z1), (t2, x2, y2, z2), ...]
        """
        df = pd.read_csv(track_points_path)
        track_data = df[
            [
                "t",                          # time stamp (s)
                "p_x", "p_y", "p_z",          # position (m)
                "v_x", "v_y", "v_z",          # linear velocity (m/s)
                "q_x", "q_y", "q_z", "q_w",   # quaternion
                "w_x", "w_y", "w_z",          # angular rate (rad/s)
                "u_1", "u_2", "u_3", "u_4",   # thrust for rotors
            ]
        ].to_numpy()

        # ensure min distance >= 1cm
        (
            self.track_timestamps,
            self.track_points,
            self.track_vel,
            self.track_quat,
            self.track_rate,
            self.track_thrust,
        ) = self.ensure_min_distance(track_data, 0.0)
        self.track_tangent = self.calculate_tangent_vectors()  # get track tangent vectors
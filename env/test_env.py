import torch
import math
import time
import yaml
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

class Test_env :
    def __init__(self, num_envs, yaml_path, device = torch.device("cuda")):

        with open(yaml_path, "r") as file:
            config = yaml.load(file, Loader = yaml.FullLoader)
        self.device = device

        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)

        self.dt = config.get("dt", 0.01)   # default sim env update in 100hz

        # create scene
        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(dt = self.dt, substeps = 2),
            viewer_options = gs.options.ViewerOptions(
                max_FPS = config.get("pid_exec_freq", 60),
                camera_pos = (3.0, 0.0, 3.0),
                camera_lookat = (0.0, 0.0, 1.0),
                camera_fov = 40,
            ),
            vis_options = gs.options.VisOptions(rendered_envs_idx = list(range(self.rendered_env_num))),
            rigid_options = gs.options.RigidOptions(
                dt = self.dt,
                constraint_solver = gs.constraint_solver.Newton,
                enable_collision = True,
                enable_joint_limit = True,
            ),
            show_viewer = True,
        )

        # add plane (ground)
        self.scene.add_entity(gs.morphs.Plane())

        # add drone
        self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        self.scene.build(n_envs = num_envs)
        
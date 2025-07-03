import torch
import math
import time
import yaml
# import taichi as ti
import genesis as gs
from utils.heapq import EntityList
from . import map
from genesis.utils.geom import trans_quat_to_T, transform_quat_by_quat
import numpy as np

class Test_env :
    def __init__(self, env_num, yaml_path, controller, imu_sim, entity, device = torch.device("cuda")):

        with open(yaml_path, "r") as file:
            config = yaml.load(file, Loader = yaml.FullLoader)
        self.device = device

        self.env_num = env_num
        self.rendered_env_num = min(10, self.env_num)

        self.dt = config.get("dt", 0.01)   # default sim env update in 100hz
        self.imu_sim = imu_sim
        self.controller = controller
        
        self.cam_quat = torch.tensor(config.get("cam_quat", [0.5, 0.5, -0.5, -0.5]), device=self.device, dtype=gs.tc_float).expand(env_num, -1)

        # create scene
        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(dt = self.dt, substeps = 1),
            viewer_options = gs.options.ViewerOptions(
                max_FPS = config.get("max_vis_FPS", 60),
                camera_pos = (-3.0, 0.0, 3.0),
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

        # creat map
        self.map = map.ForestEnv(
            min_tree_dis = 1.6, 
            width = 3, 
            length = 3
        )

        # add entity in map
        self.map.add_trees_to_scene(scene = self.scene)

        # add plane (ground)
        self.scene.add_entity(gs.morphs.Plane())

        # add drone
        self.drone = self.scene.add_entity(entity)
        setattr(self.drone, 'entity_dis_list', EntityList(max_size=config.get("max_dis_num", 5)))     # restore distance list with entity

        # follow drone
        self.scene.viewer.follow_entity(self.drone)
        if (config.get("use_FPV_camera", False)):
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(-3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )
        self.scene.build(n_envs = env_num)

    def set_FPV_cam_pos(self):
        self.cam.set_pose(
            transform = trans_quat_to_T(trans = self.drone.get_pos(), 
                                        quat = transform_quat_by_quat(self.cam_quat, self.imu_sim.body_quat))[0].cpu().numpy()
            # lookat = (0, 0, 0.5)
        )
    
    def update_entity_dis_list(self):
        cur_pos = self.drone.get_pos()
        for key, value in self.map.tree_entity_list.items():
            min_dis = self.map.get_min_dis_from_entity(value, cur_pos)
            self.drone.entity_dis_list.update(key, min_dis)

    def sim_step(self): 
        # print(f"drone pos is {self.drone.get_pos()}")
        # print(f"tree pos is {self.map.tree_entity_list[0].get_pos()}")
        # print(f"dis is {self.map.get_min_dis_from_entity(self.map.tree_entity_list[0], self.drone.get_pos())}")
        self.scene.step()

        self.update_entity_dis_list()
        self.drone.entity_dis_list.print()

        self.set_FPV_cam_pos()
        self.cam.render(rgb=True, depth=True, segmentation=False, normal=False)
        self.controller.controller_step()      # pid controller

    def get_entity(self) :
        return self.drone
    

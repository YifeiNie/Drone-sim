import torch
import math
import time
import yaml
import types
# import taichi as ti
import genesis as gs
from flight.pid import PIDcontroller
from flight.imu_sim import IMU_sim


from sensors.genesis_lidar import GenesisLidar
from sensors.LidarSensor.lidar_sensor import LidarSensor
from sensors.LidarSensor.sensor_config.lidar_sensor_config import LidarType

from flight.mavlink_sim import rc_command
from utils.heapq_ import MultiEntityList
from . import map
from genesis.utils.geom import trans_quat_to_T, transform_quat_by_quat, transform_by_trans_quat
import numpy as np

class Test_env :
    def __init__(self, env_num, yaml_path, drone, device = torch.device("cuda")):

        with open(yaml_path, "r") as file:
            self.config = yaml.load(file, Loader = yaml.FullLoader)
        self.device = device

        self.env_num = env_num
        self.rendered_env_num = min(10, self.env_num)

        self.dt = self.config.get("dt", 0.01)   # default sim env update in 100hz
        
        self.cam_quat = torch.tensor(self.config.get("cam_quat", [0.5, 0.5, -0.5, -0.5]), device=self.device, dtype=gs.tc_float).expand(env_num, -1)

        # create scene
        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(dt = self.dt, substeps = 1),
            viewer_options = gs.options.ViewerOptions(
                max_FPS = self.config.get("max_vis_FPS", 60),
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
            min_tree_dis = 1.4, 
            width = 3, 
            length = 3
        )

        # add entity in map
        self.map.add_trees_to_scene(scene = self.scene)
        
        # add plane (ground)
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # add drone
        self.drone = self.scene.add_entity(drone)
        # self.scene.viewer.follow_entity(self.drone)  # follow drone
        
        # restore distance list with entity
        setattr(self.drone, 'entity_dis_list', MultiEntityList(max_size=self.config.get("max_dis_num", 5), env_num=self.env_num))     
        
        # add imu for drone
        self.set_drone_imu()

        # add controller for drone
        self.set_drone_controller()

        # add drone camera
        self.set_drone_camera()

        # build world
        self.scene.build(n_envs = env_num)

        # add lidar
        self.set_drone_lidar()

    def update_entity_dis_list(self):
        cur_pos = self.drone.get_pos()
        for key, tree in self.map.tree_entity_list.items():
            min_dis = self.map.get_min_dis_from_entity(tree, cur_pos)
            self.drone.entity_dis_list.update(key, min_dis)

    def sim_step(self): 
        self.scene.step()

        self.update_entity_dis_list()
        self.drone.entity_dis_list.print()

        self.drone.lidar.step()

        # all_verts = []
        # entity_list = [e for e in self.scene.entities if e.idx not in [self.drone.idx, self.plane.idx]]
        # for entity in entity_list:
        #     pos = entity.get_pos()
        #     quat = entity.get_quat()
        #     for link in entity.links:
        #         for geom in link.geoms:
        #             verts = torch.tensor(geom.mesh.verts, dtype=torch.float32, device=quat.device)
        #             verts = transform_by_trans_quat(verts, pos, quat)

        #             all_verts.append(verts.detach().cpu().numpy())
                    
        # self.scene.draw_debug_spheres(
        #     poss=np.vstack(all_verts),
        #     radius=0.02,
        # )
        # self.drone.cam.set_FPV_cam_pos()
        # _, self.drone.cam.depth, _, _ = self.drone.cam.render(rgb=True, depth=True, segmentation=False, normal=False)

        self.drone.controller.step()
    
    def set_drone_imu(self):
        imu = IMU_sim(
            env_num = self.config.get("env_num", 1),
            yaml_path = "config/imu_sim_param.yaml",
            device = torch.device("cuda")
        )
        imu.set_drone(self.drone)
        setattr(self.drone, 'imu', imu) 
        
    def set_drone_lidar(self):
        lidar = GenesisLidar(
            drone=self.drone,
            scene=self.scene,
            drone_idx=[self.drone.idx, self.plane.idx],
            num_envs=1,
            headless=False,
            sensor_type=LidarType.MID360,
            visualization_mode='spheres'
        )
        # def set_lidar_pos(self):
        setattr(self.drone, 'lidar', lidar)

    def set_drone_camera(self):
        if (self.config.get("use_FPV_camera", False)):
            cam = self.scene.add_camera(
                res=(640, 480),
                pos=(-3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )
        def set_FPV_cam_pos(self):
            self.cam.set_pose(
            transform = trans_quat_to_T(trans = self.get_pos(), 
                                        quat = transform_quat_by_quat(self.cam.cam_quat, self.imu.body_quat))[0].cpu().numpy()
        )
        setattr(cam, 'cam_quat', self.cam_quat)  
        setattr(cam, 'set_FPV_cam_pos', types.MethodType(set_FPV_cam_pos, self.drone))
        depth: np.ndarray = None
        setattr(cam ,'depth', depth)
        setattr(self.drone, 'cam', cam)


    def set_drone_controller(self):
        pid = PIDcontroller(
            env_num = self.config.get("env_num", 1), 
            rc_command = rc_command,
            imu_sim = self.drone.imu, 
            yaml_path = "config/pid_param.yaml",
            device = torch.device("cuda")
        )
        pid.set_drone(self.drone)
        setattr(self.drone, 'controller', pid)      

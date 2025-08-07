import torch
import math
import time
import types
import genesis as gs
from flight.pid import PIDcontroller
from flight.odom import Odom


from sensors.genesis_lidar import GenesisLidar
from sensors.LidarSensor.lidar_sensor import LidarSensor
from sensors.LidarSensor.sensor_config.lidar_sensor_config import LidarType

from flight.mavlink_sim import rc_command
from utils.heapq_ import MultiEntityList
from env.maps.forest import ForestEnv
from genesis.utils.geom import trans_quat_to_T, transform_quat_by_quat, transform_by_trans_quat
import numpy as np

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class Genesis_env :
    def __init__(
            self, 
            env_config, 
            flight_config,
        ):
        

        self.env_config = env_config
        self.flight_config = flight_config
        self.controller = env_config["controller"]
        self.render_cam = self.env_config["render_cam"]
        self.use_rc = self.env_config["use_rc"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_envs = self.env_config.get("num_envs", 1)
        self.dt = self.env_config.get("dt", 0.01)           # default sim env update in 100hz
        self.cam_quat = torch.tensor(self.env_config.get("cam_quat", [0.5, 0.5, -0.5, -0.5]), device=self.device, dtype=gs.tc_float).expand(self.num_envs, -1)
        
        self.rendered_env_num = self.num_envs if self.render_cam else min(3, self.num_envs)
        # self.rendered_env_num = min(3, self.num_envs)
        
        # create scene
        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(dt = self.dt, substeps = 1),
            viewer_options = gs.options.ViewerOptions(
                max_FPS = self.env_config.get("max_vis_FPS", 15),
                camera_pos = (-3.0, 0.0, 3.0),
                camera_lookat = (0.0, 0.0, 1.0),
                camera_fov = 40,
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = False,
                rendered_envs_idx = list(range(self.rendered_env_num)),
                env_separate_rigid = True,
                shadow = False,
            ),
            rigid_options = gs.options.RigidOptions(
                dt = self.dt,
                constraint_solver = gs.constraint_solver.Newton,
                enable_collision = True,
                enable_joint_limit = True,
            ),
            show_viewer = self.env_config["show_viewer"],
        )

        # creat map
        self.map = ForestEnv(
            min_tree_dis = 0.7, 
            width = 3, 
            length = 3
        )

        # add entity in map
        if self.env_config["load_map"] is True:
            if self.env_config["map"] == "forest":
                self.map.add_trees_to_scene(scene = self.scene)
            elif self.env_config["map"] == "gates":
                self.map.add_gates_to_scene(scene = self.scene)

        # add plane (ground)
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # add drone
        drone = gs.morphs.Drone(
            file="assets/drone_urdf/drone.urdf", 
            pos=self.env_config["drone_init_pos"], 
            default_armature=self.flight_config.get("motor_inertia", 2.6e-07)
        )
        self.drone = self.scene.add_entity(drone)
        
        # set viewer
        if self.env_config["viewer_follow_drone"] is True:
            self.scene.viewer.follow_entity(self.drone)  # follow drone
        
        # restore distance list with entity
        self.max_size = self.env_config.get("max_dis_num", 5)
        setattr(self.drone, 'entity_dis_list', MultiEntityList(self.max_size, num_envs=self.num_envs))     
        
        # add odom for drone
        self.set_drone_imu()

        # add controller for drone
        self.set_drone_controller()

        # add drone camera
        self.set_drone_camera()

        # add target
        if self.env_config["vis_waypoints"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="assets/sphere/sphere.obj",
                    scale=0.03,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # build world
        self.scene.build(n_envs = self.num_envs)
        self.drone_init_pos = self.drone.get_pos()
        self.drone_init_quat = self.drone.get_quat()
        self.drone.set_dofs_damping(torch.tensor([0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4]))  # Set damping to a small value to avoid numerical instability

        # add lidar
        # self.set_drone_lidar()

    def step(self, action=None): 
        self.scene.step()
        self.update_entity_dis_list()
        self.drone.cam.set_FPV_cam_pos()
        if self.render_cam:
            self.drone.cam.depth = self.drone.cam.render(rgb=False, depth=True)[1]   # [1] is idx of depth img
        self.drone.controller.step(action)
        
        # self.drone.lidar.step()
        # self.get_aabb_list()
        # self.reset()

    def set_drone_imu(self):
        odom = Odom(
            num_envs = self.env_config.get("num_envs", 1),
            device = torch.device("cuda")
        )
        odom.set_drone(self.drone)
        setattr(self.drone, 'odom', odom) 
        
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
        setattr(self.drone, 'lidar', lidar)

    def set_drone_camera(self):
        if (self.env_config.get("use_FPV_camera", False)):
            cam = self.scene.add_camera(
                res=self.env_config["cam_res"],
                pos=(-3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=58,
                GUI=True,
            )
        def set_FPV_cam_pos(self):
            self.cam.set_pose(
            transform = trans_quat_to_T(trans = self.get_pos(), 
                                        quat = transform_quat_by_quat(self.cam.cam_quat, self.odom.body_quat))[0].cpu().numpy()
        )
        setattr(cam, 'cam_quat', self.cam_quat)  
        setattr(cam, 'set_FPV_cam_pos', types.MethodType(set_FPV_cam_pos, self.drone))
        depth: np.ndarray = None
        setattr(cam ,'depth', depth)
        setattr(self.drone, 'cam', cam)


    def set_drone_controller(self):
        pid = PIDcontroller(
            num_envs = self.env_config.get("num_envs", 1), 
            rc_command = rc_command,
            odom = self.drone.odom, 
            config = self.flight_config,
            device = torch.device("cuda"),
            use_rc = self.use_rc,
            controller = self.controller,
        )
        pid.set_drone(self.drone)
        setattr(self.drone, 'controller', pid)      

    def update_entity_dis_list(self):
        cur_pos = self.drone.get_pos()
        for key, tree in self.map.entity_list.items():
            min_dis = self.map.get_min_dis_from_entity(tree, cur_pos)
            self.drone.entity_dis_list.update(key, min_dis)
        # self.drone.entity_dis_list.print()

    def vis_verts(self):
        all_verts = []
        entity_list = [e for e in self.scene.entities if e.idx not in [self.drone.idx, self.plane.idx]]
        for entity in entity_list:
            pos = entity.get_pos()
            quat = entity.get_quat()
            for link in entity.links:
                for geom in link.geoms:
                    verts = torch.tensor(geom.mesh.verts, dtype=torch.float32, device=quat.device)
                    verts = transform_by_trans_quat(verts, pos, quat)
                    all_verts.append(verts.detach().cpu().numpy())
        self.scene.draw_debug_spheres(
            poss=np.vstack(all_verts),
            radius=0.02,
        )

    def get_aabb_list(self):
        """
        Get a set of bounding box vertices of occupations

        :param: none
        :return: list(torch.tensor(num_envs, 2, 3))
        """
        aabb_list = []
        for entity in self.scene.entities:
            if (entity.idx == self.plane.idx or entity.idx == self.target.idx):
                continue
            aabb_list.append(entity.get_AABB())
        return aabb_list

    def reset(self, env_idx=None):
        if len(env_idx) == 0:
            return
        if env_idx is None:
            reset_range = torch.arange(self.num_envs, device=self.device)
        else:
            reset_range = env_idx    
        init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        init_pos[:, 0] = gs_rand_float(*self.env_config["init_x_range"], (self.num_envs,), self.device)
        init_pos[:, 1] = gs_rand_float(*self.env_config["init_y_range"], (self.num_envs,), self.device)
        init_pos[:, 2] = gs_rand_float(*self.env_config["init_z_range"], (self.num_envs,), self.device)

        self.drone.set_pos(init_pos[reset_range], envs_idx=reset_range, zero_velocity=True)
        self.drone.set_quat(self.drone_init_quat[reset_range], envs_idx=reset_range, zero_velocity=True)
        self.drone.odom.reset(reset_range)
        self.drone.controller.reset(reset_range)
        # self.scene.step()
        self.update_entity_dis_list()
        

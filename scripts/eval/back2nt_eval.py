import torch
import math
import time
import types
import genesis as gs
from flight.pid import PIDcontroller
import yaml
from flight.odom import Odom

from flight.mavlink_sim import rc_command
from utils.heapq_ import MultiEntityList
from env.maps.forest import ForestEnv
from genesis.utils.geom import trans_quat_to_T, transform_quat_by_quat, transform_by_trans_quat, quat_to_R
import numpy as np
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, dim_obs=10, dim_action=6) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, 192)
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.act(img_feat + self.v_proj(v))
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, None, hx


class Genesis_env :
    def __init__(self, config, controller_config):
        self.controller_config = controller_config
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_envs = config.get("num_envs", 1)
        self.dt = self.config.get("dt", 0.01)           # default sim env update in 100hz
        self.cam_quat = torch.tensor(self.config.get("cam_quat", [0.5, 0.5, -0.5, -0.5]), device=self.device, dtype=gs.tc_float).expand(self.num_envs, -1)
        self.rendered_env_num = min(3, self.num_envs)
        self.p_target = torch.tensor([[3, 0, 1.5]], device=self.device)
        # self.margin = torch.rand((self.num_envs, ), device=self.device) * 0.2 + 0.1
        self.margin = torch.zeros((self.num_envs,), device=self.device )
        # create scene
        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(dt = self.dt, substeps = 1),
            viewer_options = gs.options.ViewerOptions(
                max_FPS = self.config.get("max_vis_FPS", 15),
                camera_pos = (-3.0, 0.0, 3.0),
                camera_lookat = (0.0, 0.0, 1.0),
                camera_fov = 40,
            ),
            vis_options = gs.options.VisOptions(
                rendered_envs_idx = list(range(self.rendered_env_num)),
                env_separate_rigid=True,
            ),
            rigid_options = gs.options.RigidOptions(
                dt = self.dt,
                constraint_solver = gs.constraint_solver.Newton,
                enable_collision = True,
                enable_joint_limit = True,
            ),
            show_viewer = True,
        )

        # creat map
        self.map = ForestEnv(
            min_tree_dis = 0.9, 
            width = 4, 
            length = 4
        )

        # add entity in map
        self.map.add_gates_to_scene(scene = self.scene)

        # add plane (ground)
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # add drone
        drone = gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(-1.5, 0.0, 0.0))
        self.drone = self.scene.add_entity(drone)
        
        # set viewer
        # self.scene.viewer.follow_entity(self.drone)  # follow drone
        
        # restore distance list with entity
        setattr(self.drone, 'entity_dis_list', MultiEntityList(max_size=self.config.get("max_dis_num", 5), num_envs=self.num_envs))     
        
        # add odom for drone
        self.set_drone_imu()

        # add controller for drone
        self.set_drone_controller()

        # add drone camera
        self.set_drone_camera()

        # add target
        if self.config["vis_waypoints"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="assets/entity_src/sphere/sphere.obj",
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


    def step(self, action=None): 
        self.scene.step()

        self.drone.cam.set_FPV_cam_pos()
        _,self.drone.cam.depth,_,_ = self.drone.cam.render(rgb=True, depth=True)   # [1] is idx of depth img
        self.drone.controller.step(action)

    def set_drone_imu(self):
        odom = Odom(
            num_envs = self.config.get("num_envs", 1),
            device = torch.device("cuda")
        )
        odom.set_drone(self.drone)
        setattr(self.drone, 'odom', odom) 

    def set_drone_camera(self):
        if (self.config.get("use_FPV_camera", False)):
            cam = self.scene.add_camera(
                res=(64, 48),
                pos=(-3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
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
            config = self.controller_config,
            num_envs = self.config.get("num_envs", 1), 
            rc_command = rc_command,
            odom = self.drone.odom, 
            device = torch.device("cuda")
        )
        pid.set_drone(self.drone)
        setattr(self.drone, 'controller', pid)      


    def reset(self, env_idx=None):
        if env_idx is None:
            reset_range = torch.arange(self.num_envs, device=self.device)
        else:
            reset_range = env_idx    

        self.drone.set_pos(self.drone_init_pos[reset_range], envs_idx=reset_range)
        self.drone.zero_all_dofs_velocity(reset_range)
        self.drone.set_quat(self.drone_init_quat[reset_range], envs_idx=reset_range)
        self.drone.odom.reset(reset_range)
        self.drone.controller.reset(reset_range)
        self.drone.odom.odom_update()
        
def model_process(model, env, h):
    num_envs = env.num_envs
    R_ = quat_to_R(env.drone.odom.body_quat)
    
    fwd = R_[:, :, 0].clone()
    up = torch.zeros_like(fwd)
    fwd[:, 2] = 0
    up[:, 2] = 1
    fwd = F.normalize(fwd, 2, -1)
    R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

    target_v_raw = env.p_target - env.drone.odom.world_pos.detach()    # tensor(num_envs, 3）
    target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
    target_v_unit = target_v_raw / target_v_norm
    target_v = target_v_unit * torch.minimum(target_v_norm, torch.tensor([[0.5]]))    # max speed
    state = [
        torch.squeeze(target_v[:, None] @ R, 1),
        R[:, 2],
        env.margin[:, None]]
    local_v = torch.squeeze(env.drone.odom.world_linear_vel[:, None] @ R, 1)
    state.insert(0, local_v)
    state = torch.cat(state, -1)
    
    dep = torch.from_numpy(env.drone.cam.depth)
    # dep = torch.abs(dep - 255)
    x = 3 / dep.clamp_(0.3, 24) - 0.6 + torch.randn_like(dep) * 0.02
    x = F.max_pool2d(x[:, None], 4, 4)
    act, values, h = model(x.to("cuda"), state, h)
    act[:, :3] = torch.bmm(act[:, :3].unsqueeze(1), R.transpose(1, 2)).squeeze(1)

    # print(act)
    return act, values, h

def acc_to_ctbr(act, num_envs=1):
    action = torch.zeros(num_envs, 4)
    action[:, 2] = -torch.atan2(act[:, 1], act[:, 0])      # yaw
    action[:, 1] = act[:, 0]* 0.5                              # pitch
    action[:, 0] = act[:, 1]* 0.5                             # roll
    action[:, -1] = -act[:, 2]                            # thr
    action = torch.tanh(action)
    action[:, -1] = (action[:, -1] + 1)*1.2
    return action


if __name__ == "__main__" :
    print("loading...")
    # logging_level="warning"
    gs.init(logging_level="warning")
    with open("config/sim_env/env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    with open("config/sim_env/flight.yaml", "r") as file:
        controller_config = yaml.load(file, Loader=yaml.FullLoader)
    model = Model()
    model.to("cuda")
    model.load_state_dict(torch.load("logs/back2nt/checkpoint0004.pth", map_location='cuda'))
    model.eval()
    genesis_env = Genesis_env(
        config = env_config, 
        controller_config = controller_config, 
    )
    genesis_env.step()      # avoid depth image None
    genesis_env.step()
    start_time = time.time()
    print("ready!")
    h = None
    while True:
        act, val, h = model_process(model, genesis_env, h)
        action = acc_to_ctbr(act)
        print(action)
        genesis_env.step(action)

        current_time = time.time()
        if current_time - start_time >= 15:
            print(f"Executed for {4} seconds, resetting.")
            genesis_env.reset()

            start_time = time.time()  
        

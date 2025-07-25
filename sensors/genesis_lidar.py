#!/usr/bin/env python3
"""
Genesis G1 Robot Environment with LidarSensor Visualization

This script demonstrates proper lidar point visualization in Genesis with G1 robot.
Key features:
1. Proper coordinate transformation handling Warp (xyzw) vs Genesis (wxyz) quaternions
2. Real-time lidar point cloud visualization
3. Multiple sensor types support
4. Clean visualization with color-coded distance information
"""
import open3d as o3d
import torch
import numpy as np
import genesis as gs
import warp as wp
from typing import Optional, Tuple, List
import os
import sys
import math

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from sensors.LidarSensor.lidar_sensor import LidarSensor
from sensors.LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType


def quat_genesis_to_warp(genesis_quat):
    """Convert Genesis quaternion (wxyz) to Warp quaternion (xyzw)"""
    # Genesis: (w, x, y, z) -> Warp: (x, y, z, w)
    if len(genesis_quat.shape) == 1:
        return torch.tensor([genesis_quat[1], genesis_quat[2], genesis_quat[3], genesis_quat[0]], 
                          device=genesis_quat.device, dtype=genesis_quat.dtype)
    else:
        return torch.stack([genesis_quat[:, 1], genesis_quat[:, 2], genesis_quat[:, 3], genesis_quat[:, 0]], dim=1)


def quat_warp_to_genesis(warp_quat):
    """Convert Warp quaternion (xyzw) to Genesis quaternion (wxyz)"""
    # Warp: (x, y, z, w) -> Genesis: (w, x, y, z)
    if len(warp_quat.shape) == 1:
        return torch.tensor([warp_quat[3], warp_quat[0], warp_quat[1], warp_quat[2]], 
                          device=warp_quat.device, dtype=warp_quat.dtype)
    else:
        return torch.stack([warp_quat[:, 3], warp_quat[:, 0], warp_quat[:, 1], warp_quat[:, 2]], dim=1)


def quat_apply_genesis(quat, vec):
    """Apply quaternion rotation using Genesis quaternion format (wxyz)"""
    # quat: (w, x, y, z), vec: (x, y, z)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Extract vector components
    vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]
    
    # Quaternion rotation formula
    # v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    qxyz = torch.stack([x, y, z], dim=-1)
    cross1 = torch.cross(qxyz, vec, dim=-1) + w.unsqueeze(-1) * vec
    cross2 = torch.cross(qxyz, cross1, dim=-1)
    result = vec + 2 * cross2
    
    return result


def quat_mul_genesis(q1, q2):
    """Multiply two quaternions in Genesis format (wxyz)"""
    # q1, q2: (w, x, y, z)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)


class GenesisLidar:
    """Genesis G1 Environment with advanced lidar visualization"""
    
    def __init__(self, 
                 drone,
                 scene,
                 drone_idx,
                 num_envs: int = 1,
                 device: str = 'cuda',
                 headless: bool = False,
                 sensor_type: LidarType = LidarType.MID360,
                 visualization_mode: str = 'spheres'):  # 'spheres' or 'lines'
        
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self.sensor_type = sensor_type
        self.visualization_mode = visualization_mode
        self.dt = 0.02
    
        
        # Initialize state
        self.episode_length = 0
        self.sim_time = 0.0
        self.lidar_update_counter = 0
        
        # Initialize components
        self.scene = None
        self.terrain = None
        self.obstacles = []
        self.lidar_sensor = None
        
        # Robot state
        self.base_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self.base_quat = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        self.base_lin_vel = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self.base_ang_vel = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        
        # Initialize base quaternion (Genesis format: wxyz)
        self.base_quat[:, 0] = 1.0  # w = 1
        
        # Lidar visualization state
        self.current_points = None
        self.current_distances = None
        self.point_colors = None
        
        # Sensor offset parameters (similar to IsaacGym version)# 10cm forward, 30cm up
        self.sensor_offset_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)  # No rotation offset (wxyz)
        self.sensor_translation = torch.tensor([0.0, 0.0, 0.436], device=self.device)
        
        # Create environment
        self.robot = None

        self.scene = scene
        self.entity_list = [e for e in scene.entities if e.idx not in drone_idx]
        
        # Initialize lidar after Genesis is ready
        print("Initializing lidar sensor...")
        self._setup_lidar_sensor()
        
        print("Initialization complete!")
    
    def update_env(self):
        all_verts = []
        all_faces = []
        vert_offset = 0

        for entity in self.entity_list:
            pos = entity.get_pos()
            quat = entity.get_quat()
            for link in entity.links:
                for geom in link.geoms:
                    verts = torch.tensor(geom.mesh.verts, dtype=torch.float32, device=quat.device)
                    faces = np.asarray(geom.mesh.faces)

                    verts = gs.utils.geom.transform_by_trans_quat(verts, pos, quat)
                    all_verts.append(verts.detach().cpu().numpy())

                    faces_shifted = faces + vert_offset
                    all_faces.append(faces_shifted)
                    vert_offset += len(verts)

        return np.vstack(all_verts), np.vstack(all_faces)
    
    def _update_robot_state(self):
        """Update robot state from Genesis"""
        if self.robot is not None:
            try:
                positions = self.robot.get_pos()
                quaternions = self.robot.get_quat()  # Genesis format: wxyz
                
                # Handle single env case
                if positions.dim() == 1:
                    positions = positions.unsqueeze(0)
                if quaternions.dim() == 1:
                    quaternions = quaternions.unsqueeze(0)
                
                self.base_pos[:] = positions
                self.base_quat[:] = quaternions  # Keep in Genesis format
                
                # Get velocities if available
                try:
                    lin_vel = self.robot.get_vel()
                    ang_vel = self.robot.get_ang()
                    
                    if lin_vel.dim() == 1:
                        lin_vel = lin_vel.unsqueeze(0)
                    if ang_vel.dim() == 1:
                        ang_vel = ang_vel.unsqueeze(0)
                    
                    self.base_lin_vel[:] = lin_vel
                    self.base_ang_vel[:] = ang_vel
                except:
                    self.base_lin_vel.zero_()
                    self.base_ang_vel.zero_()
                    
            except Exception as e:
                print(f"Warning: Robot state update failed: {e}")
                self.base_pos[:, 2] = 1.0
                self.base_quat[:, 0] = 1.0
    
    def _setup_lidar_sensor(self):
        """Setup LidarSensor with proper parameters"""
        print(f"Setting up LidarSensor: {self.sensor_type.value}")
        
        try:
            # Initialize Warp
            wp.init()
            
            # Create lidar config
            sensor_config = LidarConfig(
                sensor_type=LidarType.HORIZON,  # Choose your sensor type
                dt=0.02,  # CRITICAL: Must match simulation dt
                max_range=20.0,
                update_frequency=1.0/0.02,  # Update every simulation step
                return_pointcloud=True,
                pointcloud_in_world_frame=False,  # Get local coordinates first
                enable_sensor_noise=False,  # Disable for faster processing
            )

            
            # For grid-based sensors, set reasonable resolution
            if self.sensor_type == LidarType.SIMPLE_GRID:
                sensor_config.horizontal_line_num = 64
                sensor_config.vertical_line_num = 32
                sensor_config.horizontal_fov_deg_min = -90
                sensor_config.horizontal_fov_deg_max = 90
                sensor_config.vertical_fov_deg_min = -30
                sensor_config.vertical_fov_deg_max = 30
            
            # Create environment data for LidarSensor
            env_data = self._create_lidar_env_data()
            
            # Create LidarSensor with correct parameters
            self.lidar_sensor = LidarSensor(
                env=env_data,
                env_cfg={'sensor_noise': False},
                sensor_config=sensor_config,
                num_sensors=1,
                device=self.device
            )
            
            print("LidarSensor initialized successfully!")
            
        except Exception as e:
            print(f"LidarSensor setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.lidar_sensor = None
    
    def _create_lidar_env_data(self) -> dict:
        """Create environment data for LidarSensor"""

        vertices, faces = self.update_env()
        # vertices = np.array(vertices, dtype=np.float32)
        # faces = np.array(faces, dtype=np.int32)
        # 保存顶点
        # np.savetxt("/home/nyf/Genesis-Drones/Genesis-Drones/scene/3d_gen_py/vertices.txt", vertices, fmt="%.6f")  # 保存为浮点数

        # 保存面片索引
        # np.savetxt("/home/nyf/Genesis-Drones/Genesis-Drones/scene/3d_gen_py/faces.txt", faces, fmt="%d")          # 保存为整数索引
        vertex_tensor = torch.tensor( 
                vertices,
                device=self.device,
                requires_grad=False,
                dtype=torch.float32,
            )
        
        #if none type in vertex_tensor
        if vertex_tensor.any() is None:
            print("vertex_tensor is None")
        vertex_vec3_array = wp.from_torch(vertex_tensor,dtype=wp.vec3)        
        faces_wp_int32_array = wp.from_numpy(faces.flatten(), dtype=wp.int32,device=self.device)
                
        self.wp_meshes =  wp.Mesh(points=vertex_vec3_array,indices=faces_wp_int32_array)
        
        mesh_ids = self.mesh_ids_array = wp.array([self.wp_meshes.id], dtype=wp.uint64)
        # Calculate sensor position and orientation with proper offsets (like IsaacGym)
        sensor_quat = quat_mul_genesis(self.base_quat, self.sensor_offset_quat.unsqueeze(0).expand(self.num_envs, -1))
        sensor_pos = self.base_pos + quat_apply_genesis(self.base_quat, self.sensor_translation.unsqueeze(0).expand(self.num_envs, -1))
        
        # Convert Genesis quaternion to Warp format for sensor
        sensor_quat_warp = quat_genesis_to_warp(sensor_quat)
        
        # # Create mesh IDs (simple for now)
        # mesh_ids = torch.zeros(len(faces), dtype=torch.int32, device=self.device)
        
        return {
            'num_envs': self.num_envs,
            'sensor_pos_tensor': sensor_pos,
            'sensor_quat_tensor': sensor_quat_warp,  # Warp format
            'vertices': vertices,
            'faces': faces,
            'mesh_ids': mesh_ids
        }
    
    def step(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Step simulation and update lidar"""
        # Update robot state
        self._update_robot_state()
        
        # Update lidar sensor
        point_cloud, distances = None, None
        if self.lidar_sensor is not None:
            try:
                # Update sensor pose with proper offset calculation (like IsaacGym)
                sensor_quat = quat_mul_genesis(self.base_quat, self.sensor_offset_quat.unsqueeze(0).expand(self.num_envs, -1))
                sensor_pos = self.base_pos + quat_apply_genesis(self.base_quat, self.sensor_translation.unsqueeze(0).expand(self.num_envs, -1))
                sensor_quat_warp = quat_genesis_to_warp(sensor_quat)
                
                self.lidar_sensor.lidar_positions_tensor[:] = sensor_pos
                self.lidar_sensor.lidar_quat_tensor[:] = sensor_quat_warp
                
                # Get lidar data
                point_cloud, distances = self.lidar_sensor.update()
                
                # Store for visualization
                if point_cloud is not None:
                    self.current_points = point_cloud.clone()
                    self.current_distances = distances.clone()
                    self.lidar_update_counter += 1
                
            except Exception as e:
                print(f"Lidar update failed: {e}")
        
        # Step physics
        if self.scene is not None:
            self.sim_time += self.dt
            self.episode_length += 1
        
        # Visualize lidar points
        if self.current_points is not None and self.episode_length % 5 == 0:
            self._visualize_lidar_points()
        
        return point_cloud, distances
    
    def _visualize_lidar_points(self):
        """Visualize lidar points in Genesis scene"""
        if self.current_points is None or self.scene is None:
            return
        
        try:
            # Get points from first environment for visualization
            points = self.current_points[0]  # Shape: (num_points, 3)
            distances = self.current_distances[0].view(-1,1).squeeze()  # Shape: (num_points,)
            
            # Transform points to world coordinates using Genesis quaternion format
            # Current points are in sensor local frame, need to transform to world
            # Use proper sensor offset calculation
            base_quat_single = self.base_quat[0]
            sensor_quat = quat_mul_genesis(base_quat_single.unsqueeze(0), self.sensor_offset_quat.unsqueeze(0))[0]
            sensor_pos = self.base_pos[0] + quat_apply_genesis(base_quat_single.unsqueeze(0), self.sensor_translation.unsqueeze(0))[0]
            
            points = points.view(-1,3)
            sensor_quat = sensor_quat.repeat(points.shape[0],1)
            # Apply rotation and translation
            world_points = quat_apply_genesis(sensor_quat, points) + sensor_pos
            
            # Filter points by distance (remove invalid/far points)
            valid_mask = (distances > 0.1) & (distances < 20.0)
            if valid_mask.sum() == 0:
                return
            
            world_points = world_points[valid_mask]
            valid_distances = distances[valid_mask]
            
            # Sample points for visualization (too many points can slow down rendering)
            max_points = 20000
            if len(world_points) > max_points:
                indices = torch.randperm(len(world_points))[:max_points]
                world_points = world_points[indices]
                valid_distances = valid_distances[indices]
            
            # Generate colors based on distance
            colors = self._generate_distance_colors(valid_distances)
            # Visualize based on mode
            if self.visualization_mode == 'spheres':
                self._draw_point_spheres(world_points, colors)
            elif self.visualization_mode == 'lines':
                self._draw_point_lines(world_points, sensor_pos)
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def _generate_distance_colors(self, distances: torch.Tensor) -> List[Tuple[float, float, float, float]]:
        """Generate colors based on distance values"""
        # Normalize distances to 0-1
        min_dist, max_dist = 0.5, 8.0
        normalized = torch.clamp((distances - min_dist) / (max_dist - min_dist), 0, 1)
        
        colors = []
        for dist in normalized:
            # Color from red (close) to blue (far)
            r = 1.0 - dist.item()
            g = 0.5
            b = dist.item()
            a = 0.8
            colors.append((r, g, b, a))
        
        return colors
    
    def _draw_point_spheres(self, points: torch.Tensor, colors: List[Tuple[float, float, float, float]]):
        """Draw lidar points as colored spheres"""
        self.scene.clear_debug_objects()
        
        self.scene.draw_debug_spheres(
            poss=points,
            radius=0.02,
            color=colors[0]
        )

    def _draw_point_lines(self, points: torch.Tensor, sensor_pos: torch.Tensor):
        """Draw lidar points as lines from sensor"""
        self.scene.clear_debug_objects()
        
        # Draw lines from sensor to hit points
        for point in points:
            line_points = torch.stack([sensor_pos, point])
            self.scene.draw_debug_lines(
                poss=line_points.unsqueeze(0),
                color=(0.0, 1.0, 0.0, 0.6)
            )

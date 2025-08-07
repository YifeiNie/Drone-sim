import random
import math
import genesis as gs
import numpy as np
from genesis.options import morphs
import torch as th
import genesis.utils.geom as gu
class ForestEnv:
    def __init__(self, min_tree_dis, width, length):
        
        # remember to modify the file path
        self.strings = ["assets//gazebo-vegetation/gazebo_vegetation/models/tree_1/meshes/tree_1.obj", 
                        "assets//gazebo-vegetation/gazebo_vegetation/models/tree_7/meshes/tree_7.obj"]
        
        self.strings_convex = ["assets//gazebo-vegetation/gazebo_vegetation/models/tree_1/meshes/tree_1_convex.obj", 
                               "assets//gazebo-vegetation/gazebo_vegetation/models/tree_7/meshes/tree_7_convex.obj"]
        self.weights = [0.35, 0.65]

        if len(self.strings) != len(self.weights):
            raise ValueError("The length of the string list and weight list is inconsistent")
        
        self.width = width
        self.length = length
        self.min_tree_dis = min_tree_dis
        self.cell_size = self.min_tree_dis / math.sqrt(2)
        self.grid_width = int(self.width / self.cell_size) + 1
        self.grid_length = int(self.length / self.cell_size) + 1
        self.grid = [[None for _ in range(self.grid_length)] for _ in range(self.grid_width)]
        self.entity_list = {}

    def pick(self):
        return random.choices(
            population=range(len(self.strings)),   # 0, 1, 2, ...
            weights=self.weights,
            k=1)[0]
    
    def in_neighborhood(self, x, y):
        gx, gy = int(x / self.cell_size), int(y / self.cell_size)
        for i in range(max(gx - 2, 0), min(gx + 3, self.grid_width)):
            for j in range(max(gy - 2, 0), min(gy + 3, self.grid_length)):
                p = self.grid[i][j]
                if p is not None:
                    dx, dy = p[0] - x, p[1] - y
                    if dx * dx + dy * dy < self.min_tree_dis * self.min_tree_dis:
                        return True
        return False

    def generate_poisson_points(self, k=30):

        points = []
        process_list = []

        x0, y0 = random.uniform(0, self.width), random.uniform(0, self.length)
        points.append((x0, y0))
        process_list.append((x0, y0))
        self.grid[int(x0 / self.cell_size)][int(y0 / self.cell_size)] = (x0, y0)

        while process_list:
            x, y = process_list.pop(random.randint(0, len(process_list) - 1))
            for _ in range(k):
                r = random.uniform(self.min_tree_dis, 2 * self.min_tree_dis)
                theta = random.uniform(0, 2 * math.pi)
                new_x = x + r * math.cos(theta)
                new_y = y + r * math.sin(theta)
                if 0 <= new_x < self.width and 0 <= new_y < self.length and not self.in_neighborhood(new_x, new_y):
                    points.append((new_x, new_y))
                    process_list.append((new_x, new_y))
                    self.grid[int(new_x / self.cell_size)][int(new_y / self.cell_size)] = (new_x, new_y)

        return points

    def add_trees_to_scene(self, scene):
        
        random.random()
        positions = self.generate_poisson_points()

        for x, y in positions:
            idx = self.pick()
            tree_file = self.strings[idx]
            tree_file_for_collision = self.strings_convex[idx]

            scale = random.uniform(0.6, 1.3)
            roll = math.radians(random.uniform(0, 10))
            pitch = math.radians(random.uniform(0, 10))
            yaw = math.radians(random.uniform(0, 360))
            morph=morphs.Mesh(
                file=tree_file,
                pos=(x-0.5, y-1.5, 0.0),
                euler=(
                    90 + math.degrees(roll),  # roll
                    math.degrees(pitch),      # pitch  
                    math.degrees(yaw)         # yaw
                ),
                scale=(scale, scale, scale),
                collision=True,
                convexify=False,
                decimate=False,
                requires_jac_and_IK=False,
                fixed=True,
                parse_glb_with_trimesh=True,
                merge_submeshes_for_collision=False,
                group_by_material=False,
                visualization=True,
                use_3rd_file=tree_file_for_collision,
            )        
            entity = scene.add_entity(morph, material=gs.materials.Rigid(sdf_min_res=20, sdf_max_res=30))
            self.entity_list[entity.idx] = entity
            
    def add_gates_to_scene(self, scene):
        random.random()
        positions = self.generate_poisson_points()
        for x, y in positions:
            scale = random.uniform(0.8, 1.2)
            roll = math.radians(random.uniform(0, 5))
            pitch = math.radians(random.uniform(0, 5))
            yaw = math.radians(random.uniform(0, 5))
            morph=morphs.Mesh(
                file="assets//gate/gate_circle.obj",
                pos=(x+0.2, y-1.5, random.uniform(0.7, 1.1)),
                euler=(
                    90 + math.degrees(roll),  # roll
                    math.degrees(pitch),      # pitch  
                    math.degrees(yaw)         # yaw
                ),
                scale=(scale*0.015, scale*0.015, scale*0.015),
                collision=True,
                convexify=False,
                decimate=False,
                requires_jac_and_IK=False,
                fixed=True,
                parse_glb_with_trimesh=True,
                merge_submeshes_for_collision=False,
                group_by_material=False,
                visualization=True,
            )        
            entity = scene.add_entity(morph, material=gs.materials.Rigid(sdf_min_res=20, sdf_max_res=30))
            self.entity_list[entity.idx] = entity

    def get_min_dis_from_entity(self, entity, pos):
        # return: 1d-numpy or 0d numpy with num_envs == 1
        if len(entity.links) == 0:
            raise ValueError("Entity has no links.")

        min_sdf = None
        for geom in entity.geoms:
            pos_mesh = gu.inv_transform_by_trans_quat(pos, geom.get_pos(), geom.get_quat())
            sdf_value = geom._compute_sd(np.array(pos_mesh.cpu()))

            if min_sdf is None:
                min_sdf = sdf_value
            else:
                min_sdf = np.minimum(min_sdf, sdf_value)
        return min_sdf

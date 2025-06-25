import random
import math
import genesis as gs
import numpy as np
from genesis.options import morphs


def generate_poisson_points(width, height, min_dist, k=30):
    """生成泊松分布点集"""
    cell_size = min_dist / math.sqrt(2)
    grid_width = int(width / cell_size) + 1
    grid_height = int(height / cell_size) + 1
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]

    points = []
    process_list = []

    def in_neighborhood(x, y):
        gx, gy = int(x / cell_size), int(y / cell_size)
        for i in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for j in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                p = grid[i][j]
                if p is not None:
                    dx, dy = p[0] - x, p[1] - y
                    if dx * dx + dy * dy < min_dist * min_dist:
                        return True
        return False

    x0, y0 = random.uniform(0, width), random.uniform(0, height)
    points.append((x0, y0))
    process_list.append((x0, y0))
    grid[int(x0 / cell_size)][int(y0 / cell_size)] = (x0, y0)

    while process_list:
        x, y = process_list.pop(random.randint(0, len(process_list) - 1))
        for _ in range(k):
            r = random.uniform(min_dist, 2 * min_dist)
            theta = random.uniform(0, 2 * math.pi)
            new_x = x + r * math.cos(theta)
            new_y = y + r * math.sin(theta)
            if 0 <= new_x < width and 0 <= new_y < height and not in_neighborhood(new_x, new_y):
                points.append((new_x, new_y))
                process_list.append((new_x, new_y))
                grid[int(new_x / cell_size)][int(new_y / cell_size)] = (new_x, new_y)

    return points


def add_forest_to_scene(env, tree_file, map_size=(20.0, 20.0), tree_dist=2.0, seed=42):
    """将森林实体添加到仿真环境中"""
    random.seed(seed)
    width, height = map_size
    positions = generate_poisson_points(width, height, tree_dist)

    for x, y in positions:
        scale = random.uniform(0.7, 1.5)
        roll = math.radians(random.uniform(0, 10))
        pitch = math.radians(random.uniform(0, 10))
        yaw = math.radians(random.uniform(0, 360))

        env.scene.add_entity(
            morph=morphs.Mesh(
                file=tree_file,
                pos=(x, y, 0.0),
                euler=(
                    90 + math.degrees(roll),  # 强制正立
                    math.degrees(pitch),
                    math.degrees(yaw)
                ),
                scale=(scale, scale, scale),
                collision=False,
                convexify=False,
                decimate=False,
                requires_jac_and_IK=False,
                fixed=True,
                parse_glb_with_trimesh=False,
                merge_submeshes_for_collision=False,
                group_by_material=False,
                visualization=True
            )
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coacd
import trimesh
import open3d as o3d
import numpy as np

# ----------- 基本配置 -----------
name = "tree_7"   # 要处理的模型文件夹 / obj 文件前缀
base_dir = (
    "/home/nyf/Genesis-Drones/Genesis-Drones/scene/entity_src/"
    "gazebo-vegetation/gazebo_vegetation/models/"
)
input_file  = f"{base_dir}{name}/meshes/{name}.obj"
output_file = f"{base_dir}{name}/meshes/{name}_convex.obj"

# ----------- 1. 读取原始网格 -----------
mesh = trimesh.load(input_file, force="mesh")
print(f"[Original] vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")

origin_vis = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(mesh.vertices),
    triangles=o3d.utility.Vector3iVector(mesh.faces)
)
origin_vis.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([origin_vis], window_name="Original Mesh")

# ----------- 2. CoACD 凸分解 -----------
mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)
parts = coacd.run_coacd(
    mesh_coacd,
    threshold=0.09,
    max_convex_hull=-1,
    preprocess_mode="auto",
    preprocess_resolution=50,
    resolution=2000,
    mcts_nodes=20,
    mcts_iterations=100,
    mcts_max_depth=3,
    pca=False,
    merge=True,
    decimate=False,
    max_ch_vertex=256,
    extrude=False,
    extrude_margin=0.01,
    apx_mode="ch",
    seed=0
)

# ----------- 3. 打印并可视化每个凸部件 -----------
convex_meshes = []
total_v = total_f = 0

for i, (v, f) in enumerate(parts):
    v_cnt, f_cnt = len(v), len(f)
    print(f"[Convex {i:02d}] vertices: {v_cnt}, faces: {f_cnt}")
    total_v += v_cnt
    total_f += f_cnt

    mesh_vis = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(v),
        triangles=o3d.utility.Vector3iVector(f)
    )
    mesh_vis.paint_uniform_color(np.random.random(3))
    convex_meshes.append(mesh_vis)

print(f"[Convex ALL] vertices: {total_v}, faces: {total_f}")
o3d.visualization.draw_geometries(convex_meshes, window_name="Convex Decomposition")

# ----------- 4. 合并所有凸部件并导出 -----------
combined_vertices = []
combined_faces = []
offset = 0

for v, f in parts:
    v = np.asarray(v)
    f = np.asarray(f)
    combined_vertices.extend(v)
    combined_faces.extend(f + offset)
    offset += len(v)

combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)

if combined_mesh.is_empty:
    raise ValueError("combined mesh has no vertices or faces!")

combined_mesh.export(output_file)
print(f"[Saved] convex mesh => {output_file}")

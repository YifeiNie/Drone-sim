import open3d as o3d
import numpy as np
from stl import mesh as stl_mesh

# PLY点云路径
ply_path = '/home/nyf/test/pt.ply'
# STL输出路径
stl_path = '/home/nyf/test/test.stl'

# 1. 读取PLY点云
pcd = o3d.io.read_point_cloud(ply_path)
print("点云读取完成，显示点云")
o3d.visualization.draw_geometries([pcd], window_name='原始点云')

# 2. 估计法线
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 3. Poisson重建三角网格
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# # 4. 裁剪低密度顶点（去噪）
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

print("三角网格重建完成，显示网格")
o3d.visualization.draw_geometries([mesh], window_name='重建网格')

# 5. 转换为numpy-stl格式
faces = np.asarray(mesh.triangles)
vertices = np.asarray(mesh.vertices)
num_faces = faces.shape[0]
data = np.zeros(num_faces, dtype=stl_mesh.Mesh.dtype)
for i, f in enumerate(faces):
    for j in range(3):
        data['vectors'][i][j] = vertices[f[j], :]

# 6. 保存STL文件
stl_mesh_obj = stl_mesh.Mesh(data)
stl_mesh_obj.save(stl_path)
print(f"STL文件已保存：{stl_path}")

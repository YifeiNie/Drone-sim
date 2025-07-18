import numpy as np
import open3d as o3d

def load_and_visualize_mesh(vertex_file='~/Genesis-Drones/Genesis-Drones/scene/3d_gen_py/vertices.txt', face_file='~/Genesis-Drones/Genesis-Drones/scene/3d_gen_py/faces.txt'):
    vertices = np.loadtxt(vertex_file, dtype=np.float64)
    faces = np.loadtxt(face_file, dtype=np.int32)

    if faces.ndim == 1:
        faces = faces.reshape(-1, 3)

    print("顶点数量:", len(vertices))
    print("面片数量:", len(faces))

    assert vertices.ndim == 2 and vertices.shape[1] == 3, "顶点格式应为 (N, 3)"
    assert faces.ndim == 2 and faces.shape[1] == 3, "面片格式应为 (M, 3)"
    assert np.all(faces >= 0), "面片索引必须非负"
    assert faces.max() < len(vertices), "面片索引超出顶点范围"
    assert np.all(np.isfinite(vertices)), "顶点包含非有限值"

    # 如果面片索引是1-based，则需减1
    if faces.min() == 1:
        print("检测到面片索引从1开始，自动减1")
        faces -= 1

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.6, 0.7, 0.9])

    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([mesh, coord_frame], window_name="Mesh Viewer")

if __name__ == "__main__":
    load_and_visualize_mesh()

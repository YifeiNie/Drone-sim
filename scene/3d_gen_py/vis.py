import open3d as o3d
import os
import numpy as np

def get_obj_bbox_dims(filepath):
    mesh = o3d.io.read_triangle_mesh(filepath)
    if mesh.is_empty():
        raise ValueError("Failed to load mesh or mesh is empty.")

    vertices = np.asarray(mesh.vertices)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    size = max_bound - min_bound

    return {
        "length (X)": size[0],
        "width  (Y)": size[1],
        "height (Z)": size[2],
        "min_xyz": min_bound.tolist(),
        "max_xyz": max_bound.tolist()
    }

def visualize_models(root_path):
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".obj") or filename.endswith(".dae"):
                filepath = os.path.join(dirpath, filename)
                print(f"Loading {filepath}")
                mesh = o3d.io.read_triangle_mesh(filepath)
                # mesh.triangle_uvs = o3d.utility.Vector2dVector([])

                # ======================================================================
                vertices = np.asarray(mesh.vertices)
                print(len(mesh.vertices))
                # 设置阈值
                z_threshold = 0.0 
                filtered_vertices = vertices[vertices[:, 1] > z_threshold]
                filtered_mesh = mesh.select_by_index(np.where(vertices[:, 2] > z_threshold)[0])
                dec_mesh = filtered_mesh.simplify_quadric_decimation(target_number_of_triangles=5000)
                unfiltered_mesh = mesh.select_by_index(np.where(vertices[:, 2] <= z_threshold)[0])
                combined_mesh = dec_mesh + unfiltered_mesh
                print(len(combined_mesh.vertices))

                # ======================================================================
                if not mesh.is_empty():
                    print(get_obj_bbox_dims(filepath))
                    o3d.visualization.draw_geometries([combined_mesh])
                    


if __name__ == "__main__":
    visualize_models("/home/nyf/Genesis-Drones/Genesis-Drones/scene/entity_src/gazebo-vegetation/gazebo_vegetation/models")

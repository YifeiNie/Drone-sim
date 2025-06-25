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
                if not mesh.is_empty():
                    print(get_obj_bbox_dims(filepath))
                    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    visualize_models("/home/nyf/Genesis-Drones/scene/entity_src/gazebo-vegetation/gazebo_vegetation/models")

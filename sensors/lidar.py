import genesis as gs
import torch
import warp as wp
import numpy as np

class Lidar:

    def __init__(self, scene, drone_idx, device=torch.device("cuda")):
        self.scene = scene
        self.entity_list = [e for e in scene.entities if e.idx not in drone_idx]
        self.device = device
        self.verts = []
        self.faces = []
        self.update_env()
        

    def update_env(self):
        all_verts = []
        all_faces = []
        vert_offset = 0

        for entity in self.entity_list:
            for geom in entity.geoms:
                verts = np.asarray(geom.mesh.verts)
                faces = np.asarray(geom.mesh.faces)     # index of verts, 3 verts make a triangle

                all_verts.append(verts)

                faces_shifted = faces + vert_offset
                all_faces.append(faces_shifted)

                vert_offset += len(verts)

        self.verts = np.vstack(all_verts)  # shape: (N, 3)
        self.faces = np.vstack(all_faces)  # shape: (M, 3)

    
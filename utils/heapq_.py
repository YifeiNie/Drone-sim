import heapq
import numpy as np

class MultiEntityList:
    def __init__(self, max_size, num_envs):
        self.num_envs = num_envs
        self.max_size = max_size
        self.lists = [[] for _ in range(num_envs)]  

    def update(self, entity_idxs, new_values):
        """
        update all scenes
        entity_idxs: list/array/tensor of shape [n_envs]
        new_values: list/array/tensor of shape [n_envs]
        """
        entity_idxs = self.ensure_array(entity_idxs)
        new_values = self.ensure_array(new_values)

        for i in range(self.num_envs):
            idx = int(entity_idxs)
            val = float(new_values[i])
            self._update_single_env(i, idx, val)

    def _update_single_env(self, env_idx, entity_idx, new_value):
        heap = self.lists[env_idx]

        for i in range(len(heap)):
            if heap[i][1] == entity_idx:  
                heap[i] = [new_value, entity_idx] 
                heapq.heapify(heap)
                return
        if len(heap) >= self.max_size:
            heapq.heappop(heap)

        heapq.heappush(heap, [new_value, entity_idx])

    def print(self):
        for i, heap in enumerate(self.lists):
            print(f"Env #{i}:")
            for item in heap:
                print(f"(entity idx:{item[1]}, dis: {item[0]:.4f})")
            print("-----")

    def ensure_array(self, x):
        if np.isscalar(x):
            return np.array([x])
        x = np.asarray(x)
        if x.ndim == 0:
            return x.reshape(1)
        return x


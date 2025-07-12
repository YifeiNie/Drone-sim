import heapq
import numpy as np

class EntityIdxWithValue:
    def __init__(self, entity_idx, value):
        self.entity_idx = entity_idx
        self.value = float(value) 

    def __lt__(self, other):
        return self.value > other.value 

    def __repr__(self):
        return f"({self.entity_idx}, {self.value:.4f})"


class EntityList:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = []

    def update(self, entity_idx, new_value):
        new_value = float(new_value)
        for i, item in enumerate(self.heap):
            if item.entity_idx == entity_idx:
                self.heap[i].value = new_value
                heapq.heapify(self.heap)
                return
        if len(self.heap) >= self.max_size:
            heapq.heappop(self.heap)
        heapq.heappush(self.heap, EntityIdxWithValue(entity_idx, new_value))

    def print(self):
        for item in self.heap:
            print(item)


class MultiEntityList:
    def __init__(self, max_size, env_num):
        self.env_num = env_num
        self.lists = [EntityList(max_size) for _ in range(env_num)]

    def update(self, entity_idxs, new_values):
        """
        update all scenes
        entity_idxs: list/array/tensor of shape [n_envs]
        new_values: list/array/tensor of shape [n_envs]
        """

        for i in range(self.env_num):
            idx = int(entity_idxs)
            val = float(self.ensure_array(new_values)[i])
            self.lists[i].update(idx, val)

    def print(self):
        for i, elist in enumerate(self.lists):
            print(f"Env #{i}:")
            elist.print()
            print("-----")

    def ensure_array(self, x):
        if np.isscalar(x):
            return np.array([x])
        x = np.asarray(x)
        if x.ndim == 0: 
            return x.reshape(1)
        return x
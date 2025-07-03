
import heapq

class EntityIdxWithValue:
    def __init__(self, entity_idx, value):
        self.entity_idx = entity_idx
        self.value = value

    def __lt__(self, other):
        return self.value > other.value  

    def __repr__(self):
        return f"({self.entity_idx}, {self.value})"


class EntityList:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = []

    def update(self, entity_idx, new_value):
        for i, item in enumerate(self.heap):
            if item.entity_idx == entity_idx:
                self.heap[i].value = new_value
                heapq.heapify(self.heap)
                return

        if len(self.heap) >= self.max_size:
            heapq.heappop(self.heap)

        heapq.heappush(self.heap, EntityIdxWithValue(entity_idx, new_value))

    def print(self):
        print("Current Entities:")
        for item in self.heap:
            print(item)
import random
from collections import deque
import numpy as np
from . import sumtree

class ReplayMemory(object):
    def __init__(self, capacity,
                 use_compress=False):
        self.memory = generate_deque(use_compress, capacity)

    def push(self, data):
        """Saves a transition."""
        self.memory.append(data)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()

    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)


class PrioritizedMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.transitions = deque(maxlen=capacity)
        self.priorities = sumtree.SumTree()
    
    def push(self, transitions, priorities):
        self.transitions.extend(transitions)
        self.priorities.extend(priorities)
        
    def sample(self, batch_size):
        idxs,  prios = self.priorities.prioritized_sample(batch_size)
        return [self.transitions[i] for i in idxs], prios, idxs
    
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def remove_to_fit(self):
        if len(self.priorities) - self.capacity <= 0:
            return
        for _ in range(len(self.priorities) - self.capacity):
            self.transitions.popleft()
            self.priorities.popleft()

    def __len__(self):
        return len(self.transitions)

    def total_prios(self):
        return self.priorities.root.value
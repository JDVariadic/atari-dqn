import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.data = deque()
        self.buffer_size = buffer_size

    def insert_data(self, state, action, reward, next_state, terminated, truncated):
        self.data.append((state, action, reward, next_state, terminated, truncated))

    def sample_data(self, num_of_data):
        return random.sample(self.data, min(len(self.data), num_of_data))
    
    def update_replay_buffer(self):
        if len(self.data) > self.buffer_size:
            self.data.popleft()
    
    def get_length(self):
        return len(self.data)

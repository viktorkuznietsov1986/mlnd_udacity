import random
from collections import namedtuple, deque

class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples for replay. """
    
    def __init__(self, buffer_size, batch_size):
        
        self.memory = deque(maxlen=buffer_size) #internal memory
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        
    def add(self, state, action, reward, next_state, done):
        """Add an experience to an internal memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    
    def sample(self, batch_size = 64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
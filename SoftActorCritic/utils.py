import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def _process_state(self, state):
        if isinstance(state, (list, tuple)):
            processed = np.concatenate([np.array(s).flatten() for s in state])
        else:
            processed = np.array(state).flatten()
        return processed

    def add(self, state, action, reward, next_state, done):
        state = self._process_state(state)
        next_state = self._process_state(next_state)
        self.state[self.ptr] = state
        self.action[self.ptr] = np.array(action)
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.done[ind])
        )

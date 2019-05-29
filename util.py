import torch
import random
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action',  'reward'))


use_cuda = torch.cuda.is_available()
torch_cuda = torch.cuda if use_cuda else torch
device = torch.device('cuda' if use_cuda else 'cpu')


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        if isinstance(args[0], Transition):
            self.memory[self.position] = args[0]
        else:
            self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

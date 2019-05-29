import torch
import random
import numpy as np
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


class MovingAverage:
    def __init__(self, num):
        super().__init__()
        self.num = num
        self.items = []

    def add(self, value):
        self.items.append(float(value))
        if len(self.items) > self.num:
            self.items = self.items[-self.num:]

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

    def flush(self):
        self.items = []

    def __format__(self, mode):
        text = f'{self.mean():.5f}'
        if 's' not in mode:
            return text
        return text + f'±{self.std() * 100:.2f}%'

    def __float__(self):
        return self.mean()


class History:
    def __init__(self, file, header, resume):
        super().__init__()
        os.makedirs('history', exist_ok=True)
        self.file_name = f'history/{file}.csv'
        self.file = open(self.file_name, 'a' if resume else 'w')
        self.header = header
        if resume:
            log.debug(f'Continued history from {file!r}.')
        else:
            log.debug(f'Created history in {file!r}.')
            self.file.write(', '.join(header) + '\n')

    def record(self, info):
        self.file.write(', '.join(str(info[k]) for k in self.header) + '\n')

    def flush(self):
        self.file.flush()

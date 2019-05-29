import torch
import random
import math
import numpy as np
from torch import nn
from util import device

class BaseAgent():
    def __init__(self, eps_start, eps_end, eps_decay):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay


    def select_action(self, state):
        sample = random.random()
        sampling_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1 * self.step_cnt / self.eps_decay)
        self.step_cnt += 1
        if sample > sampling_threshold:
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.tensor(
                        state, device=device, dtype=torch.float)
                return torch.tensor(
                    [[torch.argmax(self.policy_net(state))]],
                    device=device, dtype=torch.long)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=device, dtype=torch.long)


class LinearHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        layers = [
            nn.Linear(in_features, 256),
            nn.Dropout(p=0.75)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvHead(nn.Module):
    def __init__(self, in_channels, expand=1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 16 * expand, kernel_size=3, stride=1),
            nn.BatchNorm2d(16*expand),
            nn.ReLU(),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


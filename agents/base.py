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
        self.best_reward = 0

    def sampling_annealing(self):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1 * self.step_cnt / self.eps_decay)
        self.epsilon = epsilon

    def select_action(self, state):
        self.net.eval()
        sample = random.random()
        self.sampling_annealing()
        if sample > self.epsilon:
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.tensor(
                        state, device=device, dtype=torch.float)
                state = state.unsqueeze(0)
                res = self.net(state)
                return torch.tensor(
                    [[torch.argmax(res)]],
                    device=device, dtype=torch.long)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=device, dtype=torch.long)

    def save_best(self, reward):
        if self.step_cnt % self.save_interval == 0:
            self.save()
            self.best_reward = reward


class LinearHead(nn.Module):
    def __init__(self, in_features, out_features=16):
        super().__init__()
        layers = [
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
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


def get_linear(in_features, out_features):
    # linear combo with batchnorm 1d and relu
    return [
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU()]


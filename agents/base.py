import torch
import random
import math
from torch import nn

class BaseAgent():
    def select_action(state):
        sample = random.random()
        sampling_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > sampling_threshold:
            with torch.no_grad():
                return self.policy_net(state)
        else:
            return torch.tensor(
                [[random.randange(self.n_actions)]],
                device=device, dtype=torch.long)


class LinearHead(nn.Module):
    def __init__(self, in_features):
        layers = [
            nn.Linear(in_features, 256),
            nn.Dropout(p=0.75)
        ]
        self.layers = layers

    def forward(x):
        return self.layers(x)


class ConvHead(nn.Module):
    def __init__(self, in_channels, expand=1):
        layers = [
            nn.Conv2d(in_channels, 16 * expand, kernel_size=3, stride=1),
            nn.BatchNorm2d(16*expand),
            nn.ReLU(),
        ]
        self.layers = layers

    def forward(x):
        return self.layers(x)


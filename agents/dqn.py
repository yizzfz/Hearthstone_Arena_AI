from .base import BaseAgent, LinearHead, ConvHead
from torch import nn


class DQN_Net(nn.Module):

    def __init__(self, head, n_actions, expand=1):
        super(DQN_Net, self).__init__()
        self.head = head
        layers = [
            nn.Linear(256, 512*expand),
            nn.Dropout(p=0.75),
            nn.ReLU(),
            nn.Linear(512*expand, 1024*expand),
            nn.Dropout(p=0.75),
            nn.ReLU(),
            nn.Linear(1024*expand, n_actions)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        head = self.head(x)
        return self.layers(head)


class DQN(BaseAgent):
    def __init__(self, state_shape, n_actions):
        super(DQN, self).__init__()
        if len(state_shape) <= 4:
            self.head = LinearHead(state_shape[0])
        else:
            self.head = ConvHead(state_shape)

        self.net = DQN_Net(self.head, n_actions)

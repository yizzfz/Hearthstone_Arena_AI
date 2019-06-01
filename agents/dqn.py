from .base import BaseAgent, LinearHead, ConvHead
from torch import nn
from util import Transition, MovingAverage, device, to_torch_var, to_numpy
from torch.nn import functional as F
import torch
import numpy as np
from log import log


class DQN_Net(nn.Module):
    def __init__(self, head, n_actions, expand=1):
        super(DQN_Net, self).__init__()
        self.head = head
        layers = [
            nn.Linear(256, 512*expand),
            nn.BatchNorm1d(512*expand),
            nn.ReLU(),
            nn.Linear(512*expand, 1024*expand),
            nn.BatchNorm1d(1024*expand),
            nn.ReLU(),
            nn.Linear(1024*expand, n_actions)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        head = self.head(x)
        return self.layers(head)


class DQN(BaseAgent):
    '''
    The dqn implementation largely depends on:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    https://github.com/transedward/pytorch-dqn
    '''
    def __init__(
            self, state_shape, n_actions, env_name, episodes,
            update_rate=10, eps_start=0.9,
            eps_end=0.05, eps_decay=10000, step_size=1000,
            decay_factor=0.5, lr=0.01):
        super(DQN, self).__init__(
            eps_start, eps_end, eps_decay
        )
        if len(state_shape) <= 4:
            self.head = LinearHead(state_shape[0])
        else:
            self.head = ConvHead(state_shape)

        self.episodes = episodes
        self.update_rate = update_rate
        self.n_actions = n_actions
        self.name='checkpoints/dqn_' + env_name

        self.net = DQN_Net(self.head, n_actions).to(device)
        self.actor_net = DQN_Net(self.head, n_actions).to(device)

        self.actor_net.load_state_dict(self.net.state_dict())
        self.actor_net.eval()
        self.step_cnt = 0
        self.step_size = step_size
        self.decay_factor = decay_factor
        self.lr = lr

    def get_Q_actor(self, states:np.ndarray):
        states = to_torch_var(states)
        return self.actor_net(states)

    def get_Q(self, states:np.ndarray):
        states = to_torch_var(states)
        return self.net(states)

    def train(self, batch, batch_size, gamma, time_stamp):
        self.step_cnt += 1
        if len(batch) < batch_size:
            return None
        # set up optimziers
        optimizer= torch.optim.RMSprop(
            self.net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.step_size, self.decay_factor)

        # batch = Transition(*zip(*batch))

        states = np.vstack([x.state for x in batch])
        actions = np.vstack([x.action for x in batch])
        rewards = np.vstack([x.reward for x in batch])
        next_states = np.vstack([x.next_state for x in batch])
        done = np.vstack([x.done for x in batch])
        actions = to_torch_var(actions).long()


        Q_pred = self.get_Q(states).gather(1, actions)
        done_mask = (~done).astype(np.float)
        Q_expected = np.max(to_numpy(self.get_Q_actor(next_states)), axis=1)
        Q_expected = np.expand_dims(Q_expected, axis=1)
        Q_expected = done_mask * Q_expected
        Q_target = rewards + gamma * Q_expected
        Q_target = to_torch_var(Q_target)

        loss = F.smooth_l1_loss(Q_pred, Q_target)
        optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        scheduler.step()
        if time_stamp % self.update_rate == 0:
            self.actor_net.load_state_dict(self.net.state_dict())
        return loss

    def save(self):
        state_dict = self.net.state_dict()
        torch.save(state_dict, self.name)


    def load(self, name):
        state = torch.load(name, device)
        self.net.load_state_dict(state)


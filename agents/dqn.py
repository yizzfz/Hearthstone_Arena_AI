from .base import BaseAgent, LinearHead, ConvHead, get_linear
from torch import nn
from util import Transition, MovingAverage, device, to_torch_var, to_numpy
from torch.nn import functional as F
import torch
import numpy as np
import copy
from log import log

network_structure = {
    'head': {'type': 'linear', 'out_features': 16},
    'linear1': {'in_features': 16, 'out_features': 128},
    'linear2': {'in_features': 128, 'out_features': 256},
    'linear3_final': {'in_features': 256, 'out_features': 2},
}

class DQN_Net(nn.Module):
    def __init__(self, head, n_actions, expand=2):
        super(DQN_Net, self).__init__()
        # TODO: make a flexible network defition
        self.head = head
        layers = []

        for t, args in network_structure.items():
            args = copy.deepcopy(args)
            if 'linear' in t:
                if t != 'linear1':
                    args['in_features'] *= expand
                if not 'final' in t:
                    args['out_features'] *= expand
                layers += get_linear(**args)

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
            update_rate=10, eps_start=0.99,
            eps_end=0.01, eps_decay=5000, step_size=1000,
            decay_factor=0.5, lr=0.01, save_interval=1000):
        super(DQN, self).__init__(
            eps_start, eps_end, eps_decay
        )
        if network_structure['head']['type'] == 'linear':
            self.head = LinearHead(
                in_features=state_shape[0],
                out_features=network_structure['head']['out_features'])
        else:
            self.head = ConvHead(state_shape)

        self.episodes = episodes
        self.update_rate = update_rate
        self.n_actions = n_actions
        self.name='checkpoints/dqn_' + env_name

        self.net = DQN_Net(self.head, n_actions).to(device)
        self.actor_net = DQN_Net(self.head, n_actions).to(device)

        self.actor_net.load_state_dict(self.net.state_dict())
        self.step_cnt = 0
        self.step_size = step_size
        self.decay_factor = decay_factor
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, self.step_size, self.decay_factor)
        self.lossfn = torch.nn.MSELoss()
        self.save_interval = save_interval

    def get_Q(self, states:np.ndarray):
        states = to_torch_var(states)
        return self.net(states)

    def get_actor_Q(self, states:np.ndarray):
        states = to_torch_var(states)
        return self.net(states)

    def train(self, batch, batch_size, gamma, time_stamp):
        self.net.train()
        self.step_cnt += 1
        assert(len(batch) == batch_size)
        if len(batch) < batch_size:
            return None

        states = np.vstack([x.state for x in batch])
        actions = np.vstack([x.action for x in batch])
        rewards = np.vstack([x.reward for x in batch])
        next_states = np.vstack([x.next_state for x in batch])
        done = np.vstack([x.done for x in batch])
        actions = to_torch_var(actions).long()

        Q_pred = self.get_Q(states).gather(1, actions)
        done_mask = (~done).astype(np.float)
        Q_expected = np.max(to_numpy(self.get_actor_Q(next_states)), axis=1)
        Q_expected = np.expand_dims(Q_expected, axis=1)
        Q_expected = done_mask * Q_expected
        Q_target = rewards + gamma * Q_expected
        Q_target = to_torch_var(Q_target)

        self.optimizer.zero_grad()
        loss = self.lossfn(Q_pred, Q_target)
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.scheduler.step()
        if self.step_cnt % self.update_rate == 0:
            self.actor_net.load_state_dict(self.net.state_dict())
        return loss

    def save(self):
        state_dict = self.net.state_dict()
        torch.save(state_dict, self.name)

    def load(self, name):
        state = torch.load(name, device)
        self.net.load_state_dict(state)

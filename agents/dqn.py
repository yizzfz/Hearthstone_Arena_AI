from .base import BaseAgent, LinearHead, ConvHead
from torch import nn
from util import Transition, MovingAverage, device
from torch.nn import functional as F
import torch
from log import log


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
    '''
    The dqn implementation largely depends on:
    https: // pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''
    def __init__(
            self, state_shape, n_actions, env_name, episodes,
            update_rate=10, eps_start=0.9,
            eps_end=0.05, eps_decay=200, step_size=1000, decay_factor=0.5):
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

        self.policy_net = DQN_Net(self.head, n_actions).to(device)
        self.target_net = DQN_Net(self.head, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.step_cnt = 0
        self.step_size = step_size
        self.decay_factor = decay_factor


    def train(self, batch, batch_size, gamma, time_stamp):
        self.step_cnt += 1
        if len(batch) < batch_size:
            return None
        optimizer= torch.optim.RMSprop(self.policy_net.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.step_size, self.decay_factor)
        batch = Transition(*zip(*batch))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.uint8)

        non_final_next_states = [
            torch.tensor([s], device=device, dtype=torch.float)
            for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(non_final_next_states, dim=0)
        state_np = [torch.tensor([s], device=device, dtype=torch.float) for s in batch.state]
        action_np = [torch.tensor([s], device=device, dtype=torch.long) for s in batch.action]
        reward_np = [torch.tensor([s], device=device) for s in batch.reward]

        state_batch = torch.cat(state_np)
        action_batch = torch.cat(action_np)
        reward_batch = torch.cat(reward_np)
        next_state_values = torch.zeros(batch_size, device=device)

        tmp = self.policy_net(state_batch)
        # pick state_action value using action batch
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(-1))
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (
            next_state_values * gamma) + reward_batch
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        scheduler.step()
        if time_stamp % self.update_rate == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss

    def save(self):
        state_dict = self.policy_net.state_dict()
        torch.save(state_dict, self.name)


    def load(self, name):
        state = torch.load(name, device)
        self.policy_net.load_state_dict(state)


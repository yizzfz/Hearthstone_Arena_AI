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
            update_rate=10, eps_start=0.99,
            eps_end=0.01, eps_decay=10000, step_size=1000,
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
        assert(len(batch) == batch_size)
        if len(batch) < batch_size:
            return None
        # set up optimziers
        optimizer= torch.optim.Adam(
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

        # import pdb; pdb.set_trace()
        Q_pred = self.get_Q(states).gather(1, actions)
        done_mask = (~done).astype(np.float)
        Q_expected = np.max(to_numpy(self.get_Q(next_states)), axis=1)
        Q_expected = np.expand_dims(Q_expected, axis=1)
        Q_expected = done_mask * Q_expected
        Q_target = rewards + gamma * Q_expected
        Q_target = to_torch_var(Q_target)

        loss = F.smooth_l1_loss(Q_pred, Q_target)
        # loss = torch.nn.MSELoss(Q_pred, Q_target)
        optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        scheduler.step()
        # if time_stamp % self.update_rate == 0:
        #     self.actor_net.load_state_dict(self.net.state_dict())
        return loss

    def save(self):
        state_dict = self.net.state_dict()
        torch.save(state_dict, self.name)


    def load(self, name):
        state = torch.load(name, device)
        self.net.load_state_dict(state)



# class DQN(torch.nn.Module):
#     def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
#         """DQN Network
#         Args:
#             input_dim (int): `state` dimension.
#                 `state` is 2-D tensor of shape (n, input_dim)
#             output_dim (int): Number of actions.
#                 Q_value is 2-D tensor of shape (n, output_dim)
#             hidden_dim (int): Hidden dimension in fc layer
#         """
#         super(DQN, self).__init__()

#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),
#             torch.nn.BatchNorm1d(hidden_dim),
#             torch.nn.PReLU()
#         )

#         self.layer2 = torch.nn.Sequential(
#             torch.nn.Linear(hidden_dim, hidden_dim),
#             torch.nn.BatchNorm1d(hidden_dim),
#             torch.nn.PReLU()
#         )

#         self.final = torch.nn.Linear(hidden_dim, output_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Returns a Q_value
#         Args:
#             x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
#         Returns:
#             torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
#         """
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.final(x)

#         return x




# class Agent(object):

#     def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
#         """Agent class that choose action and train
#         Args:
#             input_dim (int): input dimension
#             output_dim (int): output dimension
#             hidden_dim (int): hidden dimension
#         """
#         self.dqn = DQN(input_dim, output_dim, hidden_dim)
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         self.loss_fn = torch.nn.MSELoss()
#         self.optim = torch.optim.Adam(self.dqn.parameters())

#     def _to_variable(self, x: np.ndarray) -> torch.Tensor:
#         """torch.Variable syntax helper
#         Args:
#             x (np.ndarray): 2-D tensor of shape (n, input_dim)
#         Returns:
#             torch.Tensor: torch variable
#         """
#         return torch.autograd.Variable(torch.Tensor(x))

#     def select_action(self, states: np.ndarray, eps: float) -> int:
#         """Returns an action
#         Args:
#             states (np.ndarray): 2-D tensor of shape (n, input_dim)
#             eps (float): 𝜺-greedy for exploration
#         Returns:
#             int: action index
#         """
#         if np.random.rand() < eps:
#             return np.random.choice(self.output_dim)
#         else:
#             self.dqn.train(mode=False)
#             scores = self.get_Q(states)
#             _, argmax = torch.max(scores.data, 1)
#             return int(argmax.numpy())

#     def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
#         """Returns `Q-value`
#         Args:
#             states (np.ndarray): 2-D Tensor of shape (n, input_dim)
#         Returns:
#             torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
#         """
#         states = self._to_variable(states.reshape(-1, self.input_dim))
#         self.dqn.train(mode=False)
#         return self.dqn(states)

#     def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
#         """Computes `loss` and backpropagation
#         Args:
#             Q_pred (torch.FloatTensor): Predicted value by the network,
#                 2-D Tensor of shape(n, output_dim)
#             Q_true (torch.FloatTensor): Target value obtained from the game,
#                 2-D Tensor of shape(n, output_dim)
#         Returns:
#             float: loss value
#         """
#         self.dqn.train(mode=True)
#         self.optim.zero_grad()
#         loss = self.loss_fn(Q_pred, Q_true)
#         loss.backward()
#         self.optim.step()

#         return loss


#     def train_helper(self, minibatch, gamma) -> float:
#         """Prepare minibatch and train them
#         Args:
#             agent (Agent): Agent has `train(Q_pred, Q_true)` method
#             minibatch (List[Transition]): Minibatch of `Transition`
#             gamma (float): Discount rate of Q_target
#         Returns:
#             float: Loss value
#         """
#         states = np.vstack([x.state for x in minibatch])
#         actions = np.array([x.action for x in minibatch])
#         rewards = np.array([x.reward for x in minibatch])
#         next_states = np.vstack([x.next_state for x in minibatch])
#         done = np.array([x.done for x in minibatch])
#         # import pdb; pdb.set_trace()

#         Q_predict = self.get_Q(states)
#         Q_target = Q_predict.clone().data.numpy()
#         Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(self.get_Q(next_states).data.numpy(), axis=1) * ~done
#         Q_target = self._to_variable(Q_target)
#         return self.train(Q_predict, Q_target)
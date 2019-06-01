import gym
import torch
import numpy as np

from collections import deque
from util import Transition, device



class Game():
    # A wrapper for an openai game
    def __init__(self, name, memory=None, render=False, force_screen=False):
        self.render = render
        self.memory = memory
        self.force_screen = force_screen
        self._game_init(name)
        self.rewards = 0

    def _game_init(self, name):
        self.env = gym.make(name)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        # fix a seed
        self.env.seed(1)
        self.reset()
        if self.force_screen:
            self.state = self.get_screen()
        # else:
        #     state = self.env.observation_space.sample()
        #     state = torch.from_numpy(
        #         np.ascontiguousarray(state))
        #     self.state = state.to(device)

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        screen = torch.from_numpy(
            np.ascontiguousarray(screen))
        # Resize, and add a batch dimension (BCHW)
        screen = screen.unsqueeze(0)
        screen = screen.to(device)
        return screen

    def get_state(self, get_shape=False):
        if get_shape:
            return self.state, list(self.state.shape)
        return self.state

    def store_to_memory(self, transition):
        assert(self.memory is not None)
        if self.memory is not None:
            self.memory.push(*transition)

    def step(self, action, return_done=True):
        # step and store transition
        next_state, reward, done, _ = self.env.step(action)
        self.rewards += reward

        if done:
            reward = -1
        if self.force_screen:
            self.last_screen = self.current_screen
            self.current_screen = self.get_screen()
            next_state = self.current_screen
        
        transition = Transition(self.state, next_state, action, reward, done)
        self.store_to_memory(transition)
        # update state
        self.state = next_state
        # render if needed
        if self.render:
            self.env.render()
        if return_done:
            return transition, done
        return transition

    def reset(self):
        self.state = self.env.reset()
        self.rewards = 0
        return self.state




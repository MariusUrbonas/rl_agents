from collections import namedtuple
from abc import ABC, abstractmethod

import random
import numpy as np
import torch


Transition = namedtuple('Transition', ('state',
                                       'action',
                                       'reward',
                                       'next_state',
                                       'done'))


class BaseReplayMemory(ABC):
    def __init__(self, capacity):
        self.capacity = capacity

    @abstractmethod
    def push(self, *args):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def __len__(self):
        pass


class ReplayMemory(BaseReplayMemory):
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))
        self.memory = self.memory[-self.capacity:]
        #print(self.memory)

    def sample(self, batch_size):
        res = random.sample(self.memory, batch_size) 
        random.shuffle(res)
        return self.unpack(res)

    def unpack(self, batch):

        batch = Transition(*zip(*batch))

        state_batch = torch.cat(batch.state)
        done_batch = torch.cat(batch.done)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(BaseReplayMemory):

    def __init__(self, capacity=100000, priority_fraction=0.0):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

    def push(self, is_prior=False, *args):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = Transition(*args)
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = Transition(*args)
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def sample(self, batch_size):
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = random.sample(self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        random.shuffle(res)
        return self.unpack(res)

    def unpack(self, batch):

        batch = Transition(*zip(*batch))

        state_batch = torch.cat(batch.state)
        done_batch = torch.cat(batch.done)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)

import torch
import torch.nn.functional as F
import numpy as np

from dqn.base_dqn import BaseDQN
from utils import to_pt, EpsilonScheduler


class DoubleDQN(BaseDQN):

    def __init__(self, policy_cls, env, gamma=0.99, learning_rate=5e-4, replay_memory_capacity=10000,
                 exploration_fraction=0.1, exploration_final_eps=0.02, batch_size=32, learning_starts=1000, 
                 log_freq=100, verbose=0, train_freq=1, target_network_update=500, tau=0.01):

        super(DoubleDQN, self).__init__(policy_cls=policy_cls, env=env, gamma=gamma, learning_rate=learning_rate, 
                 exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, batch_size=batch_size, learning_starts=learning_starts, 
                 log_freq=log_freq, verbose=verbose, train_freq=train_freq, replay_memory_capacity=replay_memory_capacity)

        self.params.TARGET_NETWORK_UPDATE = target_network_update
        self.params.TAU = tau
        self.setup_models()

    def setup_models(self):
        self.policy = self.policy_cls(self.env.observation_space.shape, self.env.action_space.n).to(self.params.DEVICE)
        self.policy.train()
        self.target = self.policy_cls(self.env.observation_space.shape, self.env.action_space.n).to(self.params.DEVICE)
        self.target.train()
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer1 = torch.optim.Adam(self.policy.parameters(), lr=self.params.LEARNING_RATE)
        #self.optimizer2 = torch.optim.Adam(self.policy2.parameters(), lr=self.params.LEARNING_RATE)


    def compute_loss(self, batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        action_batch = action_batch.view(action_batch.size(0),1)

        policy_Q = self.policy(state_batch).gather(1, action_batch)

        next_policy_a = self.policy(next_state_batch).max(1)[1]

        terget_next_Q = self.target(next_state_batch).gather(1, next_policy_a.unsqueeze(1)).view(1, self.params.BATCH_SIZE)

        expected_Q = ((terget_next_Q * self.params.GAMMA) + reward_batch).squeeze()

        expected_Q[done_batch] = 0

        loss = F.smooth_l1_loss(policy_Q, expected_Q.unsqueeze(1))
        return loss


    def update(self, batch_size):
        batch = self.replay_memory.sample(self.params.BATCH_SIZE)
        loss1 = self.compute_loss(batch)

        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        for target_param, param in zip(self.target.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.params.TAU * param + (1 - self.params.TAU) * target_param)

import torch
import torch.nn.functional as F
import numpy as np

from dqn.base_dqn import BaseDQN
from utils import to_pt, EpsilonScheduler


class VanillaDQN(BaseDQN):

    def __init__(self, policy_cls, env, gamma=0.99, learning_rate=5e-4, replay_memory_capacity=10000,
                 exploration_fraction=0.1, exploration_final_eps=0.02, batch_size=32, learning_starts=1000, 
                 log_freq=100, verbose=0, train_freq=1, target_network_update=500):

        super(VanillaDQN, self).__init__(policy_cls=policy_cls, env=env, gamma=gamma, learning_rate=learning_rate, 
                 exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, batch_size=batch_size, learning_starts=learning_starts, 
                 log_freq=log_freq, verbose=verbose, train_freq=train_freq, replay_memory_capacity=replay_memory_capacity)

        self.params.TARGET_NETWORK_UPDATE = target_network_update
        self.setup_models()


    def setup_models(self):
        self.policy = self.policy_cls(self.env.observation_space.shape, self.env.action_space.n).to(self.params.DEVICE)
        self.policy.train()
        self.target = self.policy_cls(self.env.observation_space.shape, self.env.action_space.n).to(self.params.DEVICE)
        self.target.eval()
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.params.LEARNING_RATE)


    def compute_loss(self, batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        action_batch = action_batch.view(action_batch.size(0),1)

        policy_Q = self.policy(state_batch).gather(1, action_batch)
        next_target_Q = self.target(next_state_batch).max(1)[0].detach()

        expected_Q = (next_target_Q * self.params.GAMMA) + reward_batch
        expected_Q[done_batch] = 0

        loss = F.smooth_l1_loss(policy_Q, expected_Q.unsqueeze(1))
        return loss


    def update(self, batch_size):
        batch = self.replay_memory.sample(self.params.BATCH_SIZE)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()





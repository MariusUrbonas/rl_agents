from base import OffPolicyRLModel
from utils import Params
from utils import to_pt, EpsilonScheduler

import numpy as np
import torch

class BaseDQN(OffPolicyRLModel):

    def __init__(self, policy_cls, env, gamma, learning_rate, 
                 exploration_fraction, exploration_final_eps, batch_size, learning_starts, 
                 log_freq, verbose, train_freq, replay_memory_capacity):

        super(BaseDQN, self).__init__(policy_cls=policy_cls, env=env, verbose=verbose, replay_memory_capacity=replay_memory_capacity)

        self.params = Params(gamma=gamma, learning_rate=learning_rate, exploration_fraction=exploration_fraction,
                             exploration_final_eps=exploration_final_eps, batch_size=batch_size, learning_starts=learning_starts, 
                             log_freq=log_freq, verbose=verbose, train_freq=train_freq)

        self.params.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def learn(self, total_timesteps):
        i_episode = 1
        state = self.env.reset()
        eps_scheduler = EpsilonScheduler.LinearlyDecaying(total_steps=total_timesteps, 
                                                final_eps=self.params.EXPLORATION_FINAL_EPS,
                                                decay_fract=self.params.EXPLORATION_FRACTION, 
                                                warmup_steps=self.params.LEARNING_STARTS)
        self.mean_rewards, rewards, episode_reward = [], [], 0
        self.episode_rewards = []
        self.timestep_upto_episode = []
        self.mean_rew = [0]
        self.timesteps = [0]
        rews = []

        for i_timestep in range(1, total_timesteps+1):
            # Environment loop
            action = self.get_action(state, eps=eps_scheduler.val())
            next_state, reward, done, info = self.env.step(action)

            # Put your experiance into memory
            self.replay_memory.push(to_pt(np.array([state]), type='float', device=self.params.DEVICE), 
                                    to_pt(np.array([action]), device=self.params.DEVICE),
                                    to_pt(np.array([reward]), type='float', device=self.params.DEVICE),
                                    to_pt(np.array([next_state]), type='float', device=self.params.DEVICE),
                                    to_pt(np.array([done]), type='bool', device=self.params.DEVICE))

            state = next_state
            episode_reward += reward

            # Update your policy
            if len(self.replay_memory) > self.params.BATCH_SIZE and i_timestep % self.params.TRAIN_FREQ == 0:
                self.update(self.params.BATCH_SIZE)

            ## Update your target
            if i_timestep % self.params.TARGET_NETWORK_UPDATE == 0:
                self.target.load_state_dict(self.policy.state_dict())

            # Reset your environment when done
            if done:
                state = self.env.reset()
                self.episode_rewards.append(episode_reward)
                self.timestep_upto_episode.append(i_timestep)
                rewards.append(episode_reward)
                rews.append(episode_reward)
                episode_reward = 0

                if i_episode % self.params.LOG_FREQ == 0:
                    self.mean_rewards.append(sum(rewards)/len(rewards))
                    if self.verbose > 0:
                        self.print_stats(i_timestep, eps_scheduler.val(), i_episode, self.mean_rewards[-1])
                    rewards = []

                i_episode += 1

            eps_scheduler.step()


    def get_action(self, observation, eps=0.0):
        if np.random.uniform() < eps:
            return self.env.action_space.sample()

        observation = to_pt(np.array([observation]), type='float', device=self.params.DEVICE)
        with torch.no_grad():
            qvals = self.policy(observation)
        action = qvals.max(1)[1]
        return action.item()


    def eval(self, num_episodes=100):
        rewards = []
        for i_episode in range(num_episodes):
            state = self.env.reset()
            ep_reward = 0
            done = False
            while not done:
                action = self.get_action(state, eps=0.0)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                ep_reward += reward
            state = self.env.reset()
            rewards.append(ep_reward)
        return sum(rewards)/len(rewards)


    def show(self, num_timesteps=200):
        state = self.env.reset()
        for i in range(num_timesteps):
            action = self.get_action(state, eps=0)
            self.env.render()
            next_state, reward, done, info = self.env.step(action)
            state = next_state
            if done:
                state = self.env.reset()
        self.env.close()


    def print_stats(self, steps_done, eps, i_episode, mean_reward):
        out = '-'*40 + '\n'
        out += f"| % time spent exploring  | {eps:<10}|\n"
        out += f"| episodes                | {i_episode:<10}|\n"
        out += f"| mean {self.params.LOG_FREQ:>3} episode reward | {mean_reward:<10}|\n"     
        out += f"| steps                   | {steps_done:<10}|\n"
        out += "-"*40
        print(out)
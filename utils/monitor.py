import pandas as pd
from pathlib import Path
from gym.core import Wrapper
import numpy as np
import time

class Monitor(Wrapper):
    EXT = "monitor.csv"
    file_handler = None

    def __init__(self, env, filepath, exp_name):
        Wrapper.__init__(self, env=env)
        init_data = {'ep_reward': [0], 'ep_timesteps': [0], 'ep_time': [0]}
        self.monitor = pd.DataFrame(init_data)
        self.filepath = Path(filepath)
        self.exp_name = exp_name
        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        
    def reset(self, **kwargs):
        self.t_start = time.time() 
        self.rewards = []
        return self.env.reset(**kwargs)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            # Stats:
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_time = (time.time() - self.t_start)
            # Save stats 
            self.monitor = self.monitor.append({'ep_reward': ep_rew, 'ep_timesteps': ep_len, 'ep_time': ep_time},ignore_index=True)
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(ep_time)
        self.total_steps += 1
        return observation, reward, done, info
    
    def save_log(self):
        self.filepath.mkdir(parents=True, exist_ok=True) 
        fpath = str(self.filepath) + '/' + self.exp_name + '_' + Monitor.EXT
        self.monitor.to_csv(fpath, index=False)
        print(f"[Monitor log saved at {fpath}]")
              


def load_experiments(path):
    path = Path(path)           
    found = list(path.glob(f'**/*{Monitor.EXT}'))
    print(f'[Loading experiments from: {str(path)} found {len(found)}]')
    loaded = [pd.read_csv(str(found_at)) for found_at in found]
    return loaded
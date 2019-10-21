from abc import ABC, abstractmethod
from utils import ReplayMemory, PrioritizedReplayMemory

import gym


class BaseRLModel(ABC):
    """
    The base RL model
    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param policy_base: (BasePolicy) the base policy used by this method
    """

    def __init__(self, policy_cls, env, verbose=0, **kwargs):
        self.policy_cls = policy_cls
        self.verbose = verbose
        if isinstance(env, str):
            if self.verbose >= 1:
                print(f"Creating environment from the given name: {env}")
            self.env = gym.make(env)
        else:
            self.env = env


    @abstractmethod
    def learn(self, total_timesteps, log_interval):
        """
        Return a trained model.
        :param total_timesteps: (int) The total number of samples to train on
        :param log_interval: (int) The number of timesteps before logging.
        :return: (BaseRLModel) the trained model
        """
        pass


class OffPolicyRLModel(BaseRLModel):
    """
    The base class for off policy RL model
    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param replay_buffer: (ReplayBuffer) the type of replay buffer
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param policy_base: (BasePolicy) the base policy used by this method
    """


    def __init__(self, policy_cls, env, verbose=0, replay_memory_capacity=100000):
        super(OffPolicyRLModel, self).__init__(policy_cls, env, verbose=verbose)

        self.replay_memory = ReplayMemory(capacity=replay_memory_capacity)

    def set_replay_memory(self, memory):
        self.replay_memory = memory

    def use_prioritized_memory(self, capacity, priority_frac):
        self.replay_memory = PrioritizedReplayMemory(capacity=capacity, priority_fraction=priority_frac)

    @abstractmethod
    def learn(self, total_timesteps, log_interval):
        pass


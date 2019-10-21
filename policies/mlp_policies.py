import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpPolicy(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MlpPolicy, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals


class SmallMlpPolicy(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SmallMlpPolicy, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals
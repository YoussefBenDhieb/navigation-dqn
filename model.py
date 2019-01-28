import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, value_fc_units=32, advantage_fc_units = 32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.value_fc = nn.Linear(fc1_units, value_fc_units)
        self.value = nn.Linear(value_fc_units, 1)
        self.advantage_fc = nn.Linear(fc1_units, advantage_fc_units)
        self.advantage = nn.Linear(advantage_fc_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        value = F.relu(self.value_fc(x))
        value = self.value(value)
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        output = value + (advantage - torch.mean(advantage))
        return output

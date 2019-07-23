import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size):
        super(Policy, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_dim)

        self.max_action = max_action

        self.train()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.max_action * torch.tanh(self.l2(x))
        return x

    def select_action(self, state, device):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        self.eval()

        return self(state).cpu().data.numpy().flatten()

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size):
        super(DDPGCritic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

        self.train()

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = self.l2(x)
        return x


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size):
        super(TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, 1)

        self.train()

    def forward(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)

        x1 = F.relu(self.l1(xu))
        x1 = self.l2(x1)

        x2 = F.relu(self.l4(xu))
        x2 = self.l5(x2)
        return x1, x2

    def Q1(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)

        x1 = F.relu(self.l1(xu))
        x1 = self.l2(x1)
        return x1

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from rl_algos.replay_buffer import ReplayBuffer
from rl_algos.model.layernorm_actor_critic import LN_Actor as Actor, LN_DDPGCritic as Critic
#from rl_algos.model.layernorm_mlp import LN_MLP_Actor as Actor, LN_MLP_DDPGCritic as Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, plotter):
        self.actor = Actor(state_dim, action_dim, max_action, 400, 300).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, 400, 300).to(device)
        self.actor_perturbed = Actor(state_dim, action_dim, max_action, 400, 300).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim, 400, 300).to(device)
        self.critic_target = Critic(state_dim, action_dim, 400, 300).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        self.actor_perturbed.load_state_dict(self.actor.state_dict())
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass 
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def select_action(self, state, param_noise=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        self.actor.eval()

        if param_noise is not None:
            return self.actor_perturbed(state).cpu().data.numpy().flatten()
        else:
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(
                next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)

    def save(self):
        if not os.path.exists('trained_models/DDPG/'):
            os.makedirs('trained_models/DDPG/')

        print("Saving model")

        filetype = ".pt"  # pytorch model
        torch.save(self.actor.state_dict(), os.path.join(
            "./trained_models/DDPG", "actor_model" + filetype))
        torch.save(self.critic.state_dict(), os.path.join(
            "./trained_models/DDPG", "critic_model" + filetype))

    def load(self, model_path):
        actor_path = os.path.join(model_path, "actor_model.pt")
        critic_path = os.path.join(model_path, "critic_model.pt")
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
            self.actor.eval()
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
            self.critic.eval()

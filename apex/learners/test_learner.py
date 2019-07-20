from apex.model import Policy, TD3Critic
from apex.utils import evaluator

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import gym
import gym_cassie

device = torch.device("cpu")

def gym_factory(path, **kwargs):
    from functools import partial

    """
    This is (mostly) equivalent to gym.make(), but it returns an *uninstantiated* 
    environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    spec = gym.envs.registry.spec(path)
    _kwargs = spec._kwargs.copy()
    _kwargs.update(kwargs)
    
    if callable(spec._entry_point):
        cls = spec._entry_point(**_kwargs)
    else:
        cls = gym.envs.registration.load(spec._entry_point)

    return partial(cls, **_kwargs)

import ray

@ray.remote
class Learner(object):
    def __init__(self, env_fn, memory_server, max_timesteps, state_space, action_space, num_of_evaluators=2, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        # for evaluator
        self.env_fn = env_fn
        self.max_traj_len=400

        self.env = env_fn()
        self.max_timesteps = max_timesteps

        self.batch_size=batch_size
        self.discount = discount
        self.tau = tau
        self.policy_noise=policy_noise
        self.noise_clip=noise_clip
        self.policy_freq = 2    # update frequency of policy
        self.eval_freq = 50     # how many steps before each eval
        self.num_of_evaluators = num_of_evaluators

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.global_policy = Policy(self.state_dim, self.action_dim, self.max_action, hidden_size=256).to(device)
        self.actor_target = Policy(self.state_dim, self.action_dim, self.max_action, hidden_size=256).to(device)
        self.actor_perturbed = Policy(self.state_dim, self.action_dim, self.max_action, hidden_size=256).to(device)
        self.actor_target.load_state_dict(self.global_policy.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.global_policy.parameters())

        self.critic = TD3Critic(self.state_dim, self.action_dim, self.max_action, hidden_size=256).to(device)
        self.critic_target = TD3Critic(self.state_dim, self.action_dim, self.max_action, hidden_size=256).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        
        self.memory = memory_server

        self.results = []

        # counters
        self.step_count = 0
        self.policy_step_count = 0
        self.episode_count = 0
        self.eval_step_count = 0

    def get_total_timesteps(self):
        return self.step_count

    def get_global_policy(self):
        return self.global_policy, self.is_training_finished()

    def increment_step_count(self):
        self.step_count += 1        # global step count
        self.eval_step_count += 1   # time between each eval

        # increment models' step counts
        self.policy_step_count += 1     # step count between calls of updating policy and targets (TD3)

    def is_training_finished(self):
        return self.step_count >= self.max_timesteps

    def update_eval_model(self):
        if ray.get(self.memory.storage_size.remote()) < self.batch_size:
            print("not enough experience yet")
            return

        # randomly sample a mini-batch transition from memory_server
        x, y, u, r, d = ray.get(self.memory.sample.remote(self.batch_size))
        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(u).to(device)
        next_state = torch.FloatTensor(y).to(device)
        done = torch.FloatTensor(1 - d).to(device)
        reward = torch.FloatTensor(r).to(device)

        # Select action according to policy and add clipped noise
        noise = torch.FloatTensor(u).data.normal_(0, self.policy_noise).to(device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.global_policy(next_state) +
                        noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * self.discount * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(
            current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.policy_step_count >= self.policy_freq:

            # reset step count
            self.policy_step_count = 0

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.global_policy(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.global_policy.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_and_evaluate(self):

        #print("Learner at step # {}".format(self.step_count))
        self.update_eval_model()

        # Evaluate ever so often
        if self.eval_step_count >= self.eval_freq:
            # reset eval counter
            self.eval_step_count = 0

            # evaluate global policy
            self.results.append(self.evaluate(num_of_workers=self.num_of_evaluators))
            print("Eval Return at Timestep {}: {}".format(self.step_count, self.results[-1]))

            # also save
            self.save()

    def evaluate(self, trials=30, num_of_workers=2):
        # initialize evaluators
        evaluators = [evaluator.remote(self.env_fn, self.global_policy, self.env.reset(), self.max_traj_len)
                      for _ in range(num_of_workers)]

        total_rewards = 0

        for t in range(trials):
            # get result from a worker
            ready_ids, _ = ray.wait(evaluators, num_returns=1)

            # update total rewards
            total_rewards += ray.get(ready_ids[0])

            # remove ready_ids from the evaluators
            evaluators.remove(ready_ids[0])

            # start a new worker
            evaluators.append(evaluator.remote(self.env_fn, self.global_policy, self.env.reset(), self.max_traj_len))

        # return average reward
        avg_reward = total_rewards / trials
        return avg_reward

    def get_eval_model(self):
        return self.eval_model, self.is_training_finished()

    def get_results(self):
        return self.results, self.eval_freq

    def save(self):
        if not os.path.exists('trained_models/apex/'):
            os.makedirs('trained_models/apex/')

        print("Saving model")

        filetype = ".pt"  # pytorch model
        torch.save(self.global_policy.state_dict(), os.path.join(
            "./trained_models/apex", "global_policy" + filetype))
        torch.save(self.critic.state_dict(), os.path.join(
            "./trained_models/apex", "critic_model" + filetype))

    def load(self, model_path):
        actor_path = os.path.join(model_path, "global_policy.pt")
        critic_path = os.path.join(model_path, "critic_model.pt")
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.global_policy.load_state_dict(torch.load(actor_path))
            self.global_policy.eval()
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
            self.critic.eval()
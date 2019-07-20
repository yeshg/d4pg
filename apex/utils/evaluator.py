import torch
import ray

import numpy as np

device = torch.device("cpu")

@ray.remote
def evaluator(env, policy, state, max_steps):

    env = env()

    total_reward = 0
    steps = 0
    done = False

    # evaluate performance of the passed model for one episode
    while steps < max_steps and not done:
        # use model's greedy policy to predict action
        action = policy.select_action(np.array(state), device, param_noise=None)

        # take a step in the simulation
        next_state, reward, done, _ = env.step(action)

        # update state
        state = next_state

        # increment total_reward and step count
        total_reward += reward
        steps += 1

    return total_reward

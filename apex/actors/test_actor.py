from apex.model import Policy
from apex.replay import ReplayMemory

import numpy as np
import torch

import ray



@ray.remote
def Actor(env_fn, learner_id, memory_id, action_dim):

    env = env_fn()
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high[0]), hidden_size=256)
    learner_id = learner_id
    memory_id = memory_id

    start_timesteps=1000
    act_noise = 0.3
    param_noise = True

    max_traj_len = 400
    
    # initialize vars before collection loop
    obs = env.reset()
    total_timesteps = ray.get(learner_id.get_total_timesteps.remote())
    episode_num = 0
    
    #done = False # this will be updated by learner once max_timesteps is reached

    # collection loop TODO: change while True to something else
    while True:

        cassieEnv = True

        # Query learner for latest model and termination flag
        policy, training_done = ray.get(learner_id.get_global_policy.remote())

        if training_done:
            break

        obs = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        print("new episode")
        while steps < max_traj_len:

            # TODO: Implement parameter noise

            # select and perform an action
            # Select action randomly or according to policy
            if total_timesteps < start_timesteps:
                action = torch.randn(action_dim) if cassieEnv is True else env.action_space.sample()
                action = action.numpy()
            else:
                action = policy.select_action(np.array(obs), device, param_noise)
                if act_noise != 0:
                    action = (action + np.random.normal(0, act_noise,
                                                        size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

            # Perform action
            new_obs, reward, done, _ = env.step(action)
            done_bool = 1.0 if steps + 1 == max_traj_len else float(done)
            episode_reward += reward

            # Store action in replay buffer
            memory_id.add.remote((obs, new_obs, action, reward, done_bool))

            # call update from learner
            learner_id.update_and_evaluate.remote()

            # update state
            obs = new_obs

            # increment global step count
            learner_id.increment_step_count.remote()
            steps += 1

        
        # increment episode count and reset reward
        episode_num += 1
        episode_reward = 0
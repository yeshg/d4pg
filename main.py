import argparse
import time


from apex.actors import Actor
from apex.learners import Learner
from apex.replay import ReplayBuffer_remote

# Plot results
from apex.utils import VisdomLinePlotter

import gym
import gym_cassie

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

parser = argparse.ArgumentParser()
# args common for actors and learners
parser.add_argument("--env_name", default="Cassie-mimic-v0")                    # environment name
parser.add_argument("--hidden_size", default=256)

# learner specific args
parser.add_argument("--replay_size", default=1e7, type=int)                     # replay buffer size    
parser.add_argument("--max_timesteps", default=1e7, type=float)                 # Max time steps to run environment for
parser.add_argument("--batch_size", default=100, type=int)                      # Batch size for both actor and critic

# actor specific args
parser.add_argument("--num_actors", default=2, type=int)                        # Number of actors
parser.add_argument("--policy_name", default="TD3")                             # Policy name
parser.add_argument("--start_timesteps", default=1e4, type=int)                 # How many time steps purely random policy is run for

# evaluator args
parser.add_argument("--num_evaluators", default=2, type=int)                    # Number of evaluators
parser.add_argument("--viz_port", default=8098)                                 # visdom server port

args = parser.parse_args()

import ray
ray.init(num_gpus=0)

if __name__ == "__main__":
    #torch.set_num_threads(4)

    # Experiment Name
    experiment_name = "{}_{}_{}".format(args.policy_name, args.env_name, args.num_actors)
    print("Policy: {}\nEnvironment: {}\n# of Actors:{}".format(args.policy_name, args.env_name, args.num_actors))

    # Environment and Visdom Monitoring (TODO: find way to get ray remotes do visdom logging)
    env_fn = gym_factory(args.env_name)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]
    #plotter = VisdomLinePlotter(env_name=experiment_name, port=args.viz_port)

    # Create remote learner (learner will create the evaluators) and replay buffer
    memory_id = ReplayBuffer_remote.remote(args.replay_size)
    learner_id = Learner.remote(env_fn, memory_id, args.max_timesteps, obs_dim, action_dim, num_of_evaluators=args.num_evaluators)

    ray.wait([Actor.remote(env_fn, learner_id, memory_id, action_dim) for _ in range(args.num_actors)], num_returns=args.num_actors)

    start = time.time()

    results, evaluation_freq = ray.get(learner_id.get_results.remote())

    end = time.time()

    #plot_result(results, evaluation_freq, ['distributedDDPG'], num_of_collectors, num_of_evaluators, end - start)
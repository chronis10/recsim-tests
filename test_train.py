from recsim.simulator import environment
from recsim.simulator import recsim_gym
from movies_lib.sb3_wrapper import RecSimWrapper
from movies_lib.samplers import MovieDocumentSampler
from movies_lib.model import MovieUserModel
import matplotlib.pyplot as plt

from stable_baselines3 import PPO  # Using Proximal Policy Optimization, but you can choose another algorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple


def movie_watched_rating_reward(responses):
    reward = 0.0
    for response in responses:
        if response.watched:
            reward += response.rating
    return reward/(len(responses)*5)


if __name__ == "__main__":

    slate_size = 5
    num_candidates = 20
    time_budget = 10 # Time budget for the user model, interactions per episode

    user = MovieUserModel(slate_size,time_budget)
    doc = MovieDocumentSampler()

    # Initialize the environment for the movie recommendation system
    movie_env = environment.Environment(
        user,  # Use the adapted user model for movies
        doc,     # Use the adapted document sampler for movies
        num_candidates,
        slate_size,
        
        resample_documents=True  # Enable resampling of documents for each step
    )

    eval_env = environment.Environment(
        user,  # Use the adapted user model for movies
        doc,     # Use the adapted document sampler for movies
        num_candidates,
        slate_size,
        resample_documents=False  # Enable resampling of documents for each step
    )

    movie_gym_env = recsim_gym.RecSimGymEnv(movie_env, movie_watched_rating_reward)
    movie_gym_env_eval = recsim_gym.RecSimGymEnv(eval_env, movie_watched_rating_reward)

    env = RecSimWrapper(movie_gym_env)
    eval_env = RecSimWrapper(movie_gym_env_eval)


    policy_kwargs = dict(
        net_arch=[256, 256]  # Two hidden layers with 256 neurons each
    )

    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,learning_rate=0.0001)

    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                log_path='./logs/', eval_freq=500,n_eval_episodes=10,
                                deterministic=True, render=False)

    # Include the callback in the learning process
    for i in range(10):
        model.learn(total_timesteps=10000, callback=eval_callback,progress_bar=True,reset_num_timesteps=False)

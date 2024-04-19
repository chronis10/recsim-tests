import gym
from gym.spaces import Box, Discrete, Tuple
import numpy as np


class RecSimWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RecSimWrapper, self).__init__(env)
        num_genres = 10
        num_popularity = 1
        number_of_movies = 20 
        final_shape = (num_genres + num_popularity)*number_of_movies
            

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(final_shape,), dtype=np.float32)
        self.interactions = 0

    def reset(self):
        obs = self.env.reset()
        self.interactions = 0
        return self._transform_observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.interactions += 1
        return self._transform_observation(obs), reward, done, info
    
    def normalize_popularity(self,popularity, max_popularity=5):
        """ Normalize the popularity to the range 0 to 1. """
        return popularity / max_popularity

    def one_hot_encode_genre(self,genre_id, num_genres=10):
        """Returns a one-hot encoded vector for a given genre ID."""
        one_hot = np.zeros(num_genres)
        one_hot[int(genre_id)] = 1
        return one_hot

    def _transform_observation(self, obs):
        # Flatten the observation dictionary to a single array


        processed_observations = []
        for popularity,genre  in list(obs['doc'].values()):
            # One-hot encode genre
            one_hot_genre = self.one_hot_encode_genre(genre)
            # Normalize popularity
            normalized_popularity = self.normalize_popularity(popularity)
            # Concatenate one-hot genre with normalized popularity
            movie_observation = np.concatenate([one_hot_genre, [normalized_popularity]])
            processed_observations.append(movie_observation)
        return np.concatenate(processed_observations)
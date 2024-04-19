from recsim.choice_model import MultinomialLogitChoiceModel
from recsim import user
import numpy as np
from gym import spaces
from scipy import stats

class MovieUserState(user.AbstractUserState):
    """
    Represents the state of a user within a movie recommendation system. This class models user preferences
    and behaviors such as genre preferences and general movie enjoyment. It includes mechanisms to simulate
    how user satisfaction evolves based on exposure to preferred genres and overall movie experiences.
    """
    def __init__(self, memory_discount, sensitivity, innovation_stddev,
                 movie_enjoyment_mean, movie_enjoyment_stddev,
                 preferred_genre, genre_sensitivity_stddev,
                 net_genre_exposure, time_budget, observation_noise_stddev=0.1):
        ## Transition model parameters
        self.memory_discount = memory_discount
        self.sensitivity = sensitivity
        self.innovation_stddev = innovation_stddev

        ## Engagement parameters
        self.movie_enjoyment_mean = movie_enjoyment_mean
        self.movie_enjoyment_stddev = movie_enjoyment_stddev
        self.preferred_genre = preferred_genre
        self.genre_sensitivity_stddev = genre_sensitivity_stddev

        ## State variables
        self.net_genre_exposure = net_genre_exposure
        self.satisfaction = 1 / (1 + np.exp(-sensitivity * net_genre_exposure))
        self.time_budget = time_budget

        # Noise
        self._observation_noise = observation_noise_stddev

    def create_observation(self):
        """User's state is not observable."""
        clip_low, clip_high = (-1.0 / (1.0 * self._observation_noise),
                               1.0 / (1.0 * self._observation_noise))
        noise = stats.truncnorm(
            clip_low, clip_high, loc=0.0, scale=self._observation_noise).rvs()
        noisy_sat = self.satisfaction + noise
        return np.array([noisy_sat,])

    @staticmethod
    def observation_space():
        """Defines the space in which the observations exist."""
        return spaces.Box(shape=(1,), dtype=np.float32, low=-2.0, high=2.0)

    def score_document(self, doc_obs):
        """Scoring function for user's choice model based on genre match."""
        genre_match = self.preferred_genre == doc_obs[1]
        genre_influence = np.random.normal(0, self.genre_sensitivity_stddev)
        return (1 - doc_obs[0] if genre_match else doc_obs[0]) + genre_influence
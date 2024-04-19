
from recsim import document
from recsim import user
from gym import spaces
import numpy as np

class MovieDocument(document.AbstractDocument):
    def __init__(self, doc_id, popularity, genre):
        # popularity is a float representing the general popularity of the movie
        # genre is an integer representing the genre of the movie
        self.popularity = popularity
        self.genre = genre
        super(MovieDocument, self).__init__(doc_id)

    def create_observation(self):
        # Returns a numpy array containing the popularity and genre
        return np.array([self.popularity, self.genre])

    @staticmethod
    def observation_space():
        # Defines the space of observations as a box in R^2
        # Popularity ranges from 0 to 5, genre is an integer label (for simplicity assume it's 0-9)
        return spaces.Dict({
            'popularity': spaces.Box(low=0.0, high=5.0, shape=(), dtype=np.float32),
            'genre': spaces.Discrete(10)
        })
  
    def __str__(self):
        # String representation for logging purposes
        return f"Movie {self._doc_id} with popularity {self.popularity} and genre {self.genre}."

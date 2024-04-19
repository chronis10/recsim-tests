from recsim import user
from gym import spaces
import numpy as np
from recsim.choice_model import MultinomialLogitChoiceModel
from movies_lib.samplers import MovieStaticUserSampler
from movies_lib.users import MovieUserState


class MovieResponse(user.AbstractResponse):
    # The maximum possible rating a user can give to a movie.
    MAX_RATING = 5.0

    def __init__(self, watched=False, rating=0.0):
        # watched: Boolean indicating whether the movie was watched.
        # rating: Float representing the user's rating of the movie after watching, from 0.0 to 5.0.
        self.watched = watched
        self.rating = rating

    def create_observation(self):
        # Returns a dictionary with the observed user behavior.
        # 'watched' is represented as an integer (1 for watched, 0 for not watched).
        # 'rating' is an array containing the rating as a float.
        return {'watched': int(self.watched), 'rating': np.array([self.rating])}

    @classmethod
    def response_space(cls):
        # Defines the space of possible responses.
        # 'watched' is a binary discrete space (watched or not watched).
        # 'rating' is a continuous space from 0 to MAX_RATING.
        return spaces.Dict({
            'watched': spaces.Discrete(2),  # 0 or 1
            'rating': spaces.Box(
                low=0.0, 
                high=cls.MAX_RATING, 
                shape=(1,), 
                dtype=np.float32)
        })
        
class MovieUserModel(user.AbstractUserModel):
    def __init__(self, slate_size, seed=0,time_budget = 20,):
        super(MovieUserModel, self).__init__(MovieResponse, MovieStaticUserSampler(MovieUserState, seed=seed,time_budget=time_budget), slate_size)
        self.choice_model = MultinomialLogitChoiceModel({}) 

    def simulate_response(self, slate_documents):
        
        # List of empty responses
        responses = [self._response_model_ctor() for _ in slate_documents]
        # Get choice scores from the choice model.
        self.choice_model.score_documents(self._user_state, [doc.create_observation() for doc in slate_documents])
        selected_index = self.choice_model.choose_item()
        # Populate the watched item.
        self._generate_response(slate_documents[selected_index], responses[selected_index])
        return responses

    def _generate_response(self, doc, response):
        response.watched = True
        # Linear interpolation between ratings, e.g., based on genre match.
        rating_loc = (doc.popularity * self._user_state.movie_enjoyment_mean)  # Simplified for demonstration
        rating_loc *= self._user_state.satisfaction
        rating_scale = self._user_state.movie_enjoyment_stddev
        log_rating = np.random.normal(loc=rating_loc, scale=rating_scale)
        response.rating = np.clip(np.exp(log_rating), 0, 5)  # Ensure ratings are within [0, 5]

    def update_state(self, slate_documents, responses):
        for doc, response in zip(slate_documents, responses):
            if response.watched:
                innovation = np.random.normal(scale=self._user_state.innovation_stddev)
                net_genre_exposure = (self._user_state.memory_discount * self._user_state.net_genre_exposure
                                      - 2.0 * (doc.genre - 0.5)  # Assuming genre is a numeric value
                                      + innovation)
                self._user_state.net_genre_exposure = net_genre_exposure
                satisfaction = 1 / (1.0 + np.exp(-self._user_state.sensitivity * net_genre_exposure))
                self._user_state.satisfaction = satisfaction
                self._user_state.time_budget -= 1
                return
    
    def is_terminal(self):
        """Returns a boolean indicating if the session is over based on time budget."""
        return self._user_state.time_budget <= 0
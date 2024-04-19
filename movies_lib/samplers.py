from movies_lib.documents import MovieDocument
from movies_lib.users import MovieUserState
from recsim import document
from recsim import user
import numpy as np

class MovieDocumentSampler(document.AbstractDocumentSampler):
    def __init__(self, doc_ctor=MovieDocument, **kwargs):
        super(MovieDocumentSampler, self).__init__(doc_ctor, **kwargs)
        self._doc_count = 0
        
    def sample_document(self):
        doc_features = {}
        doc_features['doc_id'] = self._doc_count
        doc_features['popularity'] = np.random.uniform(0.0, 5.0)  # Popularity score between 0 and 5
        doc_features['genre'] = np.random.randint(0, 10)  # Assign a random genre label (0 to 9)
        self._doc_count += 1
        return self._doc_ctor(**doc_features)
    

class MovieStaticUserSampler(user.AbstractUserSampler):
    _state_parameters = None
    def __init__(self,
                 user_ctor=MovieUserState,
                 memory_discount=0.9,
                 sensitivity=0.01,
                 innovation_stddev=0.05,
                 movie_enjoyment_mean=4.5,
                 movie_enjoyment_stddev=0.5,
                 preferred_genre=5,  # Assigning a genre ID as an example
                 genre_sensitivity_stddev=0.1,
                 time_budget=120,
                 **kwargs):
        self._state_parameters = {
            'memory_discount': memory_discount,
            'sensitivity': sensitivity,
            'innovation_stddev': innovation_stddev,
            'movie_enjoyment_mean': movie_enjoyment_mean,
            'movie_enjoyment_stddev': movie_enjoyment_stddev,
            'preferred_genre': preferred_genre,
            'genre_sensitivity_stddev': genre_sensitivity_stddev,
            'time_budget': time_budget
        }
        super(MovieStaticUserSampler, self).__init__(user_ctor, **kwargs)

    def sample_user(self):
        # Simulating initial genre exposure to determine starting satisfaction
        starting_genre_exposure = ((self._rng.random_sample() - .5) *
                                   (1 / (1.0 - self._state_parameters['memory_discount'])))
        self._state_parameters['net_genre_exposure'] = starting_genre_exposure
        return self._user_ctor(**self._state_parameters)
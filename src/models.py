import numpy as np

class BaselineModel:
    # We need a model. This is a baseline model that returns dummy predictions.
    def __init__(self, strategy='random', constant=2.5, random_state=42):
        '''
        Input:
            strategy - strategy to use to predict the ratings
            constant - constant to use in the constant strategy
            random_state - random state to use
        '''

        if random_state is not None:
            np.random.seed(random_state)

        self.strategy = strategy
        self.constant = constant

    
    def fit(self, X, y):
        pass

    def predict(self, X):

        # Constant strategy
        # Return a constant for all the sentences
        # Note: As correlation doesn't work with constant predictions, we add some noise
        if self.strategy == 'constant':
            # Constant vector
            out = np.ones(len(X)) * self.constant

            # Add some noise
            out = out + np.random.normal(0, 0.1, len(X))

            # Clip the values to be in range
            out[out < 0] = 0
            out[out > 5] = 5
            return out
        
        # Random strategy
        # Return a random number between 0 and 5 for all the sentences (uniform distribution)
        elif self.strategy == 'random':
            return np.random.uniform(0, 5, len(X))
        

#Base class transformer, should be abstract

import numpy as np

class Transformer:

    def __init__(self, maxLag=100):
        self._maxLag=maxLag

    def transform(self, X):
        transformedX = np.copy(X)
        return transformedX

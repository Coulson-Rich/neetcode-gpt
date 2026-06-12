import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        maximum = max(z)
        total = sum(np.exp(z - maximum))
        z = [np.exp(i - maximum) / total for i in z]            

        return np.round(z, 4)
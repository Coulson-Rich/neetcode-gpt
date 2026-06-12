import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        total = sum(np.exp(z - max(z)))
        z = [np.exp(i - max(z)) / total for i in z]            

        return np.round(z, 4)
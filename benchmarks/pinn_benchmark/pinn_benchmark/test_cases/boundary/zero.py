import numpy as np


class ZeroBoundaryCondition:
    def value(self, _):
        return 0.0

    def boundary(self, x, on_boundary=1):
        return np.ones_like(x[0])

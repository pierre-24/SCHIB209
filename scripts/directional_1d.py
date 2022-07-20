from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np

from scripts.common import odeint


class Directional1D:
    """Plot directional field (and some solution) for an ODE `y' = df(x, y)`.
    """

    def __init__(self, df: Callable[[float, float], float]):
        self.df = df

    def plot(self,
             initial_values: List[Tuple[float, float]] = [],
             grid_increments: Tuple[float, float] = (.1, .1),
             grid_cmap = plt.cm.cividis,
             dx: float = 1e-2,
             graph_limits: Tuple[float, float, float, float] = (0, 0, 1, 1),
             graph_size: Tuple[float, float] = (7, 5),
             graph_labels: Tuple[str, str] = ('x', 'y')):
        """Plot a graph with the directional field and some solutions
        """

        plt.figure(figsize=graph_size)
        plt.xlim(graph_limits[0], graph_limits[2])
        plt.ylim(graph_limits[1], graph_limits[3])

        # plot some solutions
        for y0 in initial_values:
            X = np.arange(y0[0], graph_limits[2], dx)
            Y = odeint(self.df, y0, X)
            plt.plot(X, Y, label='{}'.format(y0))

        # directional field:
        x = np.arange(graph_limits[0], graph_limits[2], grid_increments[0])
        y = np.arange(graph_limits[1], graph_limits[3], grid_increments[1])
        X, Y = np.meshgrid(x, y)

        dx = np.ones(X.shape)  # assume 1 for the moment
        dy = self.df(X, Y)  # compute slope
        
        M = (np.hypot(dx, dy))  # compute norm
        M[M == 0] = 1.  # avoid divide by zero
        dx /= M  # norm
        dy /= M  # norm

        plt.quiver(X, Y, dx, dy, M, cmap=grid_cmap)
        plt.legend()
        plt.xlabel(graph_labels[0])
        plt.ylabel(graph_labels[1])
        plt.show()


# Logistic (Verhulst)
r = .75
Nf = 2
Directional1D(lambda t, N: r * N * (Nf - N)).plot(
    initial_values=[(0, -.25), (0, .25), (0, 1), (0, 3)],
    graph_limits=(-.25, -Nf, 4, 2 * Nf),
    grid_increments=(.2, .2),
    graph_labels=('t', 'N(t)')
)

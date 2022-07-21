from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np

from scripts.ode.rk4 import odeint


class Directional1D:
    """Plot directional field (and some solution) for an ODE :math:`y' = df(x, y)`.
    """

    def __init__(self, df: Callable[[float, float], float]):
        self.df = df

    def create_figure(
            self,
            initial_values: List[Tuple[float, float]] = [],
            grid_increments: Tuple[float, float] = (.1, .1),
            grid_cmap = plt.cm.cividis,
            dx: float = 1e-2,
            graph_limits: Tuple[float, float, float, float] = (0, 0, 1, 1),
            graph_size: Tuple[float, float] = (7, 5),
            graph_labels: Tuple[str, str] = ('x', 'y')
    ) -> plt.Figure:
        """Plot a graph with the directional field and some solutions
        """

        fig = plt.figure(figsize=graph_size)
        ax = fig.add_subplot()
        ax.set_xlim(graph_limits[0], graph_limits[2])
        ax.set_ylim(graph_limits[1], graph_limits[3])

        # plot some solutions
        for y0 in initial_values:
            X = np.arange(y0[0], graph_limits[2], dx)
            Y = odeint(self.df, y0, X)
            ax.plot(X, Y, label='{}'.format(y0))

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

        ax.quiver(X, Y, dx, dy, M, cmap=grid_cmap)
        ax.legend()
        ax.set_xlabel(graph_labels[0])
        ax.set_ylabel(graph_labels[1])

        return fig




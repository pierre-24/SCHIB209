from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


class PhasePortrait2D:
    """Draw phase portrait for ODE :math:`\\vec z' = A\\vec z`.
    For real eigenvalues, also plot the corresponding lines.
    """

    def __init__(self, A: np.ndarray):
        self.A = A

    def df(self, z: np.ndarray) -> np.ndarray:
        return self.A.dot(z)

    def create_figure(
            self,
            grid_increments: Tuple[float, float] = (.1, .1),
            stream_density: float = .75,
            stream_color: str = 'red',
            eig_color: str = 'blue',
            graph_limits: Tuple[float, float, float, float] = (0, 0, 1, 1),
            graph_size: Tuple[float, float] = (7, 5),
            graph_labels: Tuple[str, str] = ('z₁', 'z₂')
    ) -> plt.Figure:

        fig = plt.figure(figsize=graph_size)
        ax = fig.add_subplot()
        ax.set_xlim(graph_limits[0], graph_limits[2])
        ax.set_ylim(graph_limits[1], graph_limits[3])

        ax.set_xlabel(graph_labels[0])
        ax.set_ylabel(graph_labels[1])

        x = np.arange(graph_limits[0], graph_limits[2], grid_increments[0])

        # eigenvectors
        ax.plot([0], [0], '-o', color=eig_color)

        lamb, eta = np.linalg.eig(self.A)

        if not issubclass(type(lamb[0]), np.complex):
            box = {
                'facecolor': 'white'
            }

            yv = eta[:, 0][1] / eta[:, 0][0] * x
            ax.plot(x, yv, color=eig_color)
            ax.text(
                eta[:, 0][0] * graph_limits[2] * .75,
                eta[:, 0][1] * graph_limits[3] * .75,
                'λ₁={}'.format(lamb[0]),
                horizontalalignment='center',
                verticalalignment='center',
                bbox=box)

            if lamb[0] != lamb[1]:
                yv = eta[:, 1][1] / eta[:, 1][0] * x
                ax.plot(x, yv, color=eig_color)
                ax.text(
                    eta[:, 1][0] * graph_limits[2] * .75,
                    eta[:, 1][1] * graph_limits[3] * .75,
                    'λ₂={}'.format(lamb[1]),
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox=box)

        # stream
        y = np.arange(graph_limits[1], graph_limits[3], grid_increments[1])
        X, Y = np.meshgrid(x, y)

        dx = np.zeros(X.shape)
        dy = np.zeros(X.shape)

        for i in range(len(x)):
            for j in range(len(y)):
                dx[j, i], dy[j, i] = self.df(np.array([x[i], y[j]]).T)

        ax.streamplot(X, Y, dx, dy, density=stream_density, color=stream_color, linewidth=.75)

        return fig
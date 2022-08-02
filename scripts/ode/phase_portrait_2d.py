from typing import Tuple

import numpy

from scripts.ode.integrator import type_df

import matplotlib.pyplot as plt
import numpy as np


class PhasePortrait2D:
    """Phase portrait for 2D ODE problem of the form :math:`\\dot\\vec u = \\vec F (\\vec u).`"""

    def __init__(self, df: type_df):
        self.df = df

    def df(self, x, Y):
        return self.df(x, Y)

    def create_figure(
            self,
            grid_increments: Tuple[float, float] = (.1, .1),
            stream_density: float = .75,
            stream_color: str = 'red',
            graph_limits: Tuple[float, float, float, float] = (0, 0, 1, 1),
            graph_size: Tuple[float, float] = (7, 5),
            graph_labels: Tuple[str, str] = ('x', 'y'),
            subplot_set: Tuple[int, int, int] = (1, 1, 1)
    ) -> plt.Figure:

        fig = plt.figure(figsize=graph_size)
        ax = fig.add_subplot(*subplot_set)
        ax.set_xlim(graph_limits[0], graph_limits[2])
        ax.set_ylim(graph_limits[1], graph_limits[3])

        ax.set_xlabel(graph_labels[0])
        ax.set_ylabel(graph_labels[1])

        x = np.arange(graph_limits[0], graph_limits[2], grid_increments[0])

        # stream
        y = np.arange(graph_limits[1], graph_limits[3], grid_increments[1])
        X, Y = np.meshgrid(x, y)

        dx = np.zeros(X.shape)
        dy = np.zeros(X.shape)

        for i in range(len(x)):
            for j in range(len(y)):
                dx[j, i], dy[j, i] = self.df(0, numpy.array([x[i], y[j]]))

        ax.streamplot(X, Y, dx, dy, density=stream_density, color=stream_color, linewidth=.75)

        return fig


class PhasePortraitAutonomous2D(PhasePortrait2D):
    """Draw phase portrait for ODE :math:`\\vec z' = A\\vec z`.
    For real eigenvalues, also plot the corresponding lines.
    """

    def __init__(self, A: np.ndarray):
        self.A = A
        super().__init__(lambda x, Y: self.A.dot(Y))

    def create_figure(
            self,
            grid_increments: Tuple[float, float] = (.1, .1),
            stream_density: float = .75,
            stream_color: str = 'red',
            eig_color: str = 'blue',
            graph_limits: Tuple[float, float, float, float] = (0, 0, 1, 1),
            graph_size: Tuple[float, float] = (7, 5),
            graph_labels: Tuple[str, str] = ('z₁', 'z₂'),
            subplot_set: Tuple[int, int, int] = (1, 1, 1)
    ) -> plt.Figure:

        fig = super().create_figure(
            grid_increments,
            stream_density,
            stream_color,
            graph_limits,
            graph_size,
            graph_labels
        )

        ax = fig.get_axes()[0]

        # add eigenvectors
        x = np.arange(graph_limits[0], graph_limits[2], grid_increments[0])
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

        return fig
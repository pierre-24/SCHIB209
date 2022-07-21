from typing import Callable, Tuple, List

import matplotlib.pyplot as plt
import numpy
from scipy import optimize


class PhasePortraitAutonomous1D:
    """Draw phase portrait for autonomous ODE :math:`y' = df(y)`.
    Also adds roots and arrows if provided estimates.
    """

    def __init__(self, df: Callable[[float], float]):
        self.df = df

    def create_figure(
            self,
            dx: float = 1e-2,
            zeros_estimate: List[float] = [],
            arrow_size: float = .03,
            arrow_sep: float = .1,
            graph_limits: Tuple[float, float, float, float] = (0, 0, 1, 1),
            graph_size: Tuple[float, float] = (7, 5),
            graph_labels: Tuple[str, str] = ('x', 'y')
    ) -> plt.Figure:

        fig = plt.figure(figsize=graph_size)
        ax = fig.add_subplot()
        ax.set_xlim(graph_limits[0], graph_limits[2])
        ax.set_ylim(graph_limits[1], graph_limits[3])

        # phase itself
        X = numpy.arange(graph_limits[0], graph_limits[2], dx)
        Y = self.df(X)
        ax.plot(X, Y)

        # horizontal line
        plt.axhline(y=0, color='gray', linestyle='--')

        # root
        if len(zeros_estimate):
            # add dots
            ax.plot(zeros_estimate, numpy.zeros(len(zeros_estimate)), 'ro')

            # add arrow
            for x in zeros_estimate:
                xi, xn = x + arrow_sep, x + arrow_sep + 2 * arrow_size
                if self.df(x+dx) < 0:
                    xi, xn = xn, xi

                p = plt.Polygon(numpy.array([[xi, -arrow_size], [xi, arrow_size], [xn, 0]]), color='red')
                ax.add_patch(p)

                xi, xn = x - arrow_sep - 2 * arrow_size, x - arrow_sep
                if self.df(x-dx) < 0:
                    xi, xn = xn, xi

                p = plt.Polygon(numpy.array([[xi, -arrow_size], [xi, arrow_size], [xn, 0]]), color='red')
                ax.add_patch(p)

        ax.set_xlabel(graph_labels[0])
        ax.set_ylabel(graph_labels[1])

        return fig

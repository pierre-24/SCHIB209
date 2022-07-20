from typing import Callable, Tuple, List

import matplotlib.pyplot as plt
import numpy
from scipy import optimize


class PhaseSpace1D:
    """Draw phase space for autonomous ODE `y' = df(y)`.
    Also adds roots and arrows if provided estimates.
    """

    def __init__(self, df: Callable[[float], float]):
        self.df = df

    def plot(self,
             dx: float = 1e-1,
             zeros_estimate: List[float] = [],
             arrow_size: float = .1,
             graph_limits: Tuple[float, float, float, float] = (0, 0, 1, 1),
             graph_size: Tuple[float, float] = (7, 5),
             graph_labels: Tuple[str, str] = ('x', 'y')):

        plt.figure(figsize=graph_size)
        plt.xlim(graph_limits[0], graph_limits[2])
        plt.ylim(graph_limits[1], graph_limits[3])

        # phase itself
        X = numpy.arange(graph_limits[0], graph_limits[2], dx)
        Y = self.df(X)
        plt.plot(X, Y)

        # horizontal line
        plt.axhline(y=0, color='gray', linestyle='--')

        # root
        if len(zeros_estimate):
            roots = optimize.root(self.df, zeros_estimate)
            print(roots.message)

            # add dots
            plt.plot(roots.x, numpy.zeros(len(zeros_estimate)), 'ro')

            # add arrow
            for x in roots.x:
                xi, xn = x + arrow_size, x + 2 * arrow_size
                if self.df(x+dx) < 0:
                    xi, xn = xn, xi

                p = plt.Polygon(numpy.array([[xi, -arrow_size], [xi, arrow_size], [xn, 0]]), color='red')
                plt.gca().add_patch(p)

                xi, xn = x - 2 * arrow_size, x - arrow_size
                if self.df(x-dx) < 0:
                    xi, xn = xn, xi

                p = plt.Polygon(numpy.array([[xi, -arrow_size], [xi, arrow_size], [xn, 0]]), color='red')
                plt.gca().add_patch(p)

        plt.xlabel(graph_labels[0])
        plt.ylabel(graph_labels[1])
        plt.show()


# Logistic (Verhulst)
r = .75
Nf = 2
PhaseSpace1D(lambda N: r * N * (Nf - N)).plot(
    graph_limits=(-1, -1, 3, 1.5),
    graph_labels=('t', 'N\'(t)'),
    zeros_estimate=[-1, 3]
)
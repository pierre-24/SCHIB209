from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy
import numpy as np


class FourierSeriesPlot:
    """Plot a Fourier series of a function defined on [-L,L]
    """

    def __init__(
            self,
            L: float,
            f: Callable[[numpy.ndarray], numpy.ndarray],
            a_0: float = .0,
            a_k: Callable[[numpy.ndarray], numpy.ndarray] = lambda k: numpy.zeros(k.shape),
            b_k: Callable[[numpy.ndarray], numpy.ndarray] = lambda k: numpy.zeros(k.shape)
    ):
        self.L = L
        self.f = f
        self.a_0 = a_0
        self.a_k = a_k
        self.b_k = b_k

    def create_figure(
            self,
            N: List[int],
            dx: float = 1e-2,
            graph_limits: Tuple[float, float, float, float] = (-10, -1, 10, 1),
            graph_size: Tuple[float, float] = (7, 5),
            graph_labels: Tuple[str, str] = ('x', 'y(x)'),
            sol_label: str = 'y(x)',
    ) -> plt.Figure:
        """Plot a graph with the directional field and some solutions
        """

        fig = plt.figure(figsize=graph_size)
        ax = fig.add_subplot()
        ax.set_xlim(graph_limits[0], graph_limits[2])
        ax.set_ylim(graph_limits[1], graph_limits[3])

        # plot Fourier approx.
        Nmax = max(N)

        a_k = numpy.zeros(Nmax + 1)
        a_k[0] = self.a_0
        a_k[1:] = self.a_k(numpy.arange(1, Nmax + 1))
        b_k = numpy.zeros(Nmax + 1)
        b_k[1:] = self.b_k(numpy.arange(1, Nmax + 1))

        X = np.arange(graph_limits[0], graph_limits[2], dx)

        Xp_sin = numpy.zeros((Nmax + 1, X.shape[0]))
        Xp_cos = numpy.zeros((Nmax + 1, X.shape[0]))

        for k in range(0, Nmax + 1):
            Xp_cos[k] = numpy.cos(k * X)
            Xp_sin[k] = numpy.sin(k * X)

        for kmax in N:
            ax.plot(
                X,
                a_k[:kmax + 1].dot(Xp_cos[:kmax + 1]) + b_k[:kmax + 1].dot(Xp_sin[:kmax + 1]),
                '-',
                label='N={}'.format(kmax)
            )

        # plot solution on [-L,L]
        X = np.arange(-self.L, self.L, dx)
        ax.plot(X, self.f(X), '-r', label=sol_label, linewidth=2)

        ax.legend()
        ax.set_xlabel(graph_labels[0])
        ax.set_ylabel(graph_labels[1])

        return fig




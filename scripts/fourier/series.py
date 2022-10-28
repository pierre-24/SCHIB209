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

    def compute_series(self, X, Nmax: int):
        a_k = numpy.zeros(Nmax + 1)
        a_k[0] = self.a_0
        a_k[1:] = self.a_k(numpy.arange(1, Nmax + 1))
        b_k = numpy.zeros(Nmax + 1)
        b_k[1:] = self.b_k(numpy.arange(1, Nmax + 1))

        Y_cos = numpy.zeros((Nmax + 1, X.shape[0]))
        Y_sin = numpy.zeros((Nmax + 1, X.shape[0]))

        for k in range(0, Nmax + 1):
            Y_cos[k] = numpy.cos(k * X)
            Y_sin[k] = numpy.sin(k * X)

        return a_k, b_k, Y_cos, Y_sin

    def create_figure(
            self,
            N: List[int],
            dx: float = 1e-2,
            graph_limits: Tuple[float, float, float, float] = (-10, -1, 10, 1),
            graph_size: Tuple[float, float] = (7, 5),
            graph_labels: Tuple[str, str] = ('x', 'y(x)'),
            sol_label: str = 'y(x)',
    ) -> plt.Figure:
        """Plot a graph with the approximate and exact results
        """

        fig = plt.figure(figsize=graph_size)
        ax = fig.add_subplot()
        ax.set_xlim(graph_limits[0], graph_limits[2])
        ax.set_ylim(graph_limits[1], graph_limits[3])

        # plot Fourier approx.
        X = np.arange(graph_limits[0], graph_limits[2], dx)
        a_k, b_k, Y_cos, Y_sin = self.compute_series(X, max(N))

        for kmax in N:
            ax.plot(
                X,
                a_k[:kmax + 1].dot(Y_cos[:kmax + 1]) + b_k[:kmax + 1].dot(Y_sin[:kmax + 1]),
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

    def create_details_figure(
            self,
            Nmax: int = 5,
            dx: float = 1e-2,
            graph_size: Tuple[float, float] = (8, 5),
            graph_labels: Tuple[str, str] = ('x', 'y(x)'),
    ) -> plt.Figure:
        """Display the details of the solutions
        """

        fig = plt.figure(figsize=graph_size)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xlim(-self.L, self.L)

        ax1.set_xlabel(graph_labels[0])
        ax1.set_ylabel(graph_labels[1])

        X = np.arange(-self.L, self.L, dx)
        a_k, b_k, Y_cos, Y_sin = self.compute_series(X, Nmax)

        # plot solution on [-L,L] + constituents
        Y = self.f(X)
        mi, mx = Y.min(), Y.max()
        l = mx - mi

        ax1.set_ylim(mi / l, Nmax + mx / l)

        ax1.plot(X, Y / l, '-r', linewidth=2)

        for i in range(Nmax + 1):
            ax1.plot(X, i + (a_k[i] * Y_cos[i] + b_k[i] * Y_sin[i]) / l, '-b')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_xlim(-.3, Nmax + .3)
        ax2.set_ylim(min(a_k.min(), b_k.min()) -.1, max(a_k.max(), b_k.max()) + .1)

        ax2.set_xlabel('k')
        ax2.set_ylabel('|c_k|')

        if not all(a_k == 0):
            ax2.stem(numpy.arange(0, Nmax + 1) - .05, a_k, '-b', label='a_k', markerfmt=' ', basefmt="gray")

        if not all(b_k == 0):
            ax2.stem(numpy.arange(0, Nmax + 1) + .05, b_k, '-g', label='b_k', markerfmt=' ', basefmt="gray")

        ax2.legend()

        return fig




import argparse

import numpy
import numpy as np

import matplotlib.pyplot as plt


def fourier_x2_series(X: numpy.ndarray, n: int = 2, L: float = 3) -> float:
    """Fourier sine serie with
    b_k = 2*L * (-1)**k / k*pi
    """

    coefs = np.zeros(n + 1)
    coefs[0] = 0

    for i in range(1, n + 1):
        coefs[i] = 2*L*(-1)**i / (i * numpy.pi)

    Xp = numpy.zeros((n + 1, X.shape[0]))
    for i in range(1, n + 1):
        Xp[i] = numpy.sin(i*X * numpy.pi / L)

    return coefs.dot(Xp)


def solution(X: np.ndarray, L: float = 3) -> np.ndarray:
    return - (numpy.mod(X + L, 2 * L) - L)


DX = 0.01
WIN = (-10, -5, 10, 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # PLOT
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot()
    ax1.set_xlim(WIN[0], WIN[2])
    ax1.set_ylim(WIN[1], WIN[3])

    X = np.arange(WIN[0], WIN[2], DX)
    ax1.plot(X, solution(X), label='Solution exacte', linewidth=2)

    for i in range(0, 5):
        ax1.plot(X, fourier_x2_series(X, 5*i), '-', label='N={}'.format(5*i))

    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y(x)')

    fig.show()

    # SAVE
    if args.save:
        fig.savefig('{}_fourier.pdf'.format(args.save))

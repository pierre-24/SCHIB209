import argparse

import numpy
import numpy as np

import matplotlib.pyplot as plt


def series(X: numpy.ndarray, n: int = 2) -> float:
    """
    a_0 = 1
    a_(n+1) = - a_n / (n+1), except for n = 2
    """

    coefs = np.zeros(n + 1)
    coefs[0] = 1

    for i in range(1, n + 1):
        if i == 2:
            coefs[i] = (1-coefs[i-1]) / i
        else:
            coefs[i] = - coefs[i-1] / i

    Xp = numpy.zeros((n + 1, X.shape[0]))
    for i in range(0, n + 1):
        Xp[i] = X**i

    return coefs.dot(Xp)


def solution(X: np.ndarray) -> np.ndarray:
    return 2 * numpy.exp(-X) + X - 1


Y0 = 1
DX = 0.25
WIN = (-.25, .5, 5, 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # PLOT
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot()
    ax1.set_xlim(WIN[0], WIN[2])
    ax1.set_ylim(WIN[1], WIN[3])

    X = np.arange(0, WIN[2], .01 * DX)
    ax1.plot(X, solution(X), label='Solution exacte', linewidth=2)

    X = np.arange(0, WIN[2], DX)
    for i in range(1, 6):
        ax1.plot(X, series(X, 2*i), '-o', label='N={}'.format(2 * i))

    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y(x)')

    fig.show()

    # SAVE
    if args.save:
        fig.savefig('{}_series.pdf'.format(args.save))

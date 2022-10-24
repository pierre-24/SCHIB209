import argparse

import numpy
import numpy as np

import matplotlib.pyplot as plt


def fourier_x2_series(X: numpy.ndarray, n: int = 2) -> float:
    """Fourier cosine serie with
    a_0 = pi**2/3
    a_k = 4*(-1)**k / k**2
    """

    coefs = np.zeros(n + 1)
    coefs[0] = np.pi**2/3

    for i in range(1, n + 1):
        coefs[i] = 4*(-1)**i / (i ** 2)

    Xp = numpy.zeros((n + 1, X.shape[0]))
    for i in range(0, n + 1):
        Xp[i] = numpy.cos(i*X)

    return coefs.dot(Xp)


def solution(X: np.ndarray) -> np.ndarray:
    return (numpy.mod(X + numpy.pi, 2 * numpy.pi)-numpy.pi)**2


DX = 0.01
WIN = (-10, -1, 10, 10)


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
        ax1.plot(X, fourier_x2_series(X, i), '-', label='N={}'.format(i))

    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y(x)')

    fig.show()

    # SAVE
    if args.save:
        fig.savefig('{}_fourier.pdf'.format(args.save))

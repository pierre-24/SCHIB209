import argparse

import numpy
import numpy as np

import matplotlib.pyplot as plt


def fourier_x2_series(X: numpy.ndarray, n: int = 2, L: float = 3) -> float:
    """Fourier serie with
    a_0 = pi / 4
    a_k = ((-1)**k - 1) / (k**2*pi)
    b_k = - (-1)**k / (k**2*pi)
    """

    coefs_cos = np.zeros(n + 1)
    coefs_cos[0] = numpy.pi / 4
    coefs_sin = np.zeros(n + 1)
    coefs_sin[0] = 0

    for k in range(1, n + 1):
        coefs_cos[k] = ((-1)**k - 1) / (k**2*numpy.pi)
        coefs_sin[k] = - (-1)**k / k

    Xp_sin = numpy.zeros((n + 1, X.shape[0]))
    Xp_cos = numpy.zeros((n + 1, X.shape[0]))

    for k in range(0, n + 1):
        Xp_cos[k] = numpy.cos(k*X)
        Xp_sin[k] = numpy.sin(k*X)

    return coefs_sin.dot(Xp_sin) + coefs_cos.dot(Xp_cos)


def solution(X: np.ndarray) -> np.ndarray:
    Y = numpy.zeros(X.shape)
    Y[:] = numpy.where(
        numpy.mod(X + numpy.pi, 2 * numpy.pi) > numpy.pi,
        numpy.mod(X, numpy.pi),
        0)
    return Y


DX = 0.01
WIN = (-10, -.5, 10, 5)


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

import argparse

import numpy

import matplotlib.pyplot as plt


def damped_spring_f(X: numpy.ndarray, m: float = 1, k: float = 1, g: float = 0) -> numpy.ndarray:
    """Explicit solution for damped spring"""

    r = g**2-4*m*k
    u = -g / (2 * m)

    if r < 0:
        t = numpy.sqrt(-r) / (2 * m)
        return numpy.exp(u * X) * (numpy.cos(t * X) + numpy.sin(t * X))

    elif r == 0:
        print('critical', g)
        e = numpy.exp(u * X)
        return e + X * e

    else:
        t = numpy.sqrt(r) / (2 * m)
        return numpy.exp((u + t) * X) + numpy.exp((u - t) * X)


DX = 0.25
WIN = (0, -2, 5, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # PLOT
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot()
    ax1.set_xlim(WIN[0], WIN[2])
    ax1.set_ylim(WIN[1], WIN[3])

    X = numpy.arange(0, WIN[2], .01 * DX)

    ax1.plot(X, damped_spring_f(X * 2 * numpy.pi), label='γ=0')
    ax1.plot(X, damped_spring_f(X * 2 * numpy.pi, g=2), label='γ=2 (cas 1)')
    ax1.plot(X, damped_spring_f(X * 2 * numpy.pi, g=3), label='γ=3 (cas 2)')
    ax1.plot(X, damped_spring_f(X * 2 * numpy.pi, g=.2), label='γ=0.2 (cas 3)')

    ax1.legend()
    ax1.set_xlabel('t / 2π')
    ax1.set_ylabel('y(t)')

    fig.show()

    # SAVE
    if args.save:
        fig.savefig('{}_spring.pdf'.format(args.save))

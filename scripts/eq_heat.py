import argparse

import numpy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as anim


DX = 0.001
WIN = (-.1, -1, 1.1, 10.1)


def u_slope(X: numpy.ndarray, t: float, N: int = 200, k: float = 0.2, L: float = 1) -> numpy.ndarray:
    return numpy.array([
        2 * L / (n * numpy.pi) * numpy.sin(n * numpy.pi * X / L) * numpy.exp(-k * (n * numpy.pi / L)**2 * t)
        for n in range(1, N+1)
    ]).sum(axis=0)


def u_step(X: numpy.ndarray, t: float, N: int = 200, k: float = 0.2, L: float = 1, Ti: float = 1) -> numpy.ndarray:
    return numpy.array([
        Ti * 4 / (n * numpy.pi) * numpy.sin(n * numpy.pi * X / L) * numpy.exp(-k * (n * numpy.pi / L)**2 * t)
        for n in range(1, N+1, 2)
    ]).sum(axis=0)


def u_step_T(
        X: numpy.ndarray,
        t: float, N: int = 200,
        k: float = 0.1,
        L: float = 1,
        Ti: float = 3.,
        T1: float = .5,
        T2: float = 10
) -> numpy.ndarray:
    return T1 + (T2-T1)/L * X + numpy.array([
        2*((Ti-T1)-(Ti-T2)*(-1)**n) / (n*numpy.pi) * numpy.sin(n * numpy.pi * X / L) * numpy.exp(-k * (n * numpy.pi / L)**2 * t)
        for n in range(1, N+1)
    ]).sum(axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # PLOT
    fig = plt.figure(figsize=(8, 5))

    ax = fig.add_subplot()
    X = np.arange(0, 1, DX)
    line, = ax.plot([], [])

    def data_gen():
        yield from numpy.arange(0, 1, 0.01)

    def init():
        ax.set_xlim(WIN[0], WIN[2])
        ax.set_ylim(WIN[1], WIN[3])
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')

    def run(t):
        t /= 200
        if t == 0:
            t = 0.001

        fig.suptitle('t={:.2f}s'.format(t))
        line.set_data(X, u_step_T(X, t))

    ani = anim.FuncAnimation(fig, run, frames=400, init_func=init, repeat_delay=1000)

    # SAVE
    if args.save:
        ani.save('{}_heat.mp4'.format(args.save), writer='ffmpeg', fps=30)
    else:
        plt.show()

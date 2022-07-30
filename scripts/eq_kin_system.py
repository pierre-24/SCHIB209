import argparse
import numpy

import matplotlib.pyplot as plt

from scripts.ode.integrator import odeint


def system_autonomous_df(K: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray:
    return K.dot(Y)


def kin_successive_df(t: float, Y: numpy.ndarray, k_1: float = .2, k_2: float = .5) -> numpy.ndarray:
    return system_autonomous_df(numpy.array([
        [-k_1, 0, 0],
        [k_1, -k_2, 0],
        [0, k_2, 0]
    ]), Y)


def kin_competitive_df(t: float, Y: numpy.ndarray, k_1: float = .2, k_2: float = .5) -> numpy.ndarray:
    return system_autonomous_df(numpy.array([
        [-k_1-k_2, 0, 0],
        [k_1, 0, 0],
        [k_2, 0, 0]
    ]), Y)


def kin_eq_df(t: float, Y: numpy.ndarray, k_1: float = .2, k_m1: float = .1) -> numpy.ndarray:
    return system_autonomous_df(numpy.array([
        [-k_1, k_m1],
        [k_1, -k_m1]
    ]), Y)


Y0 = 1
DX = 0.1
WIN = (0, 0, 15, 1.1)


def prepare_plot(ax1):
    ax1.set_xlim(WIN[0], WIN[2])
    ax1.set_ylim(WIN[1], WIN[3])
    ax1.set_xlabel('t')
    ax1.set_ylabel('C(t)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    fig_eq = plt.figure(figsize=(8, 5))
    ax1 = fig_eq.add_subplot()
    prepare_plot(ax1)

    X = numpy.arange(0, WIN[2], DX)
    Y = odeint(
        kin_eq_df,
        (0, numpy.array([Y0, 0]).transpose()),
        X
    )

    ax1.plot(X, Y[:, 0], label='A(t)')
    ax1.plot(X, Y[:, 1], label='B(t)')
    ax1.legend()

    fig_eq.show()

    # Competitive & successive
    fig_sc = plt.figure(figsize=(9, 5))
    ax1 = fig_sc.add_subplot(1, 2, 1)
    prepare_plot(ax1)

    Y = odeint(
        kin_successive_df,
        (0, numpy.array([Y0, 0, 0]).transpose()),
        X
    )
    ax1.plot(X, Y[:, 0], label='A(t)')
    ax1.plot(X, Y[:, 1], label='B(t)')
    ax1.plot(X, Y[:, 2], label='C(t)')
    ax1.legend()

    ax2 = fig_sc.add_subplot(1, 2, 2)
    prepare_plot(ax2)

    Y = odeint(
        kin_competitive_df,
        (0, numpy.array([Y0, 0, 0]).T),
        X
    )

    ax2.plot(X, Y[:, 0], label='A(t)')
    ax2.plot(X, Y[:, 1], label='B(t)')
    ax2.plot(X, Y[:, 2], label='C(t)')
    ax2.legend()

    fig_sc.show()

    # SAVE
    if args.save:
        fig_eq.savefig('{}_kin_eq.pdf'.format(args.save))
        fig_sc.savefig('{}_kin_sc.pdf'.format(args.save))


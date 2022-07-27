import argparse
import numpy as np

from scripts.ode.integrator import odeint, euler

import matplotlib.pyplot as plt


def saturation_df(t: float, N: float, r=.25) -> float:
    """
    Simple logistic equation `:math:`\\frac{dN}{dt} = N\\,(1 - N)`.
    """
    return r*(1 - N)


X_LIM = 25
Y0 = .1
SUBWIN = (1.5, .4, 4, .7)


def mk_plots(ax):
    for dx in [8, 6, 1, .1, .01]:
        X = np.arange(0, 2*X_LIM, dx)
        Y = odeint(saturation_df, (0, Y0), X, integrator=euler)
        ax.plot(X, Y, label='Δx={}'.format(dx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # PLOT
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(0, X_LIM)
    ax1.set_ylim(0, 2)

    mk_plots(ax1)

    ax1.axhline(1, color='gray', linestyle='--')

    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y(x)')

    p = plt.Polygon(
        np.array([(SUBWIN[0], SUBWIN[1]), (SUBWIN[2], SUBWIN[1]), (SUBWIN[2], SUBWIN[3]), (SUBWIN[0], SUBWIN[3])]),
        ec='gray', fill=False)

    ax1.add_patch(p)

    ax2 = fig.add_subplot(1, 2, 2)

    ax2.set_xlim(SUBWIN[0], SUBWIN[2])
    ax2.set_ylim(SUBWIN[1], SUBWIN[3])

    mk_plots(ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y(x)')

    plt.subplots_adjust(wspace=.3)

    fig.show()

    # SAVE
    if args.save:
        fig.savefig('{}_euler.pdf'.format(args.save))

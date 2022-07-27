import argparse
import numpy as np

from scripts.ode.integrator import odeint, euler, heun

import matplotlib.pyplot as plt


def saturation_df(x: float, y: float) -> float:
    """
    A sine line
    """
    return np.sin(x) * y - 1


Y0 = 1
DX = 0.5
WIN = (-.25, -35, 15, 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # PLOT
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot()
    ax1.set_xlim(WIN[0], WIN[2])
    ax1.set_ylim(WIN[1], WIN[3])

    X = np.arange(0, 1.25 * WIN[2], DX)

    Y = odeint(saturation_df, (0, Y0), X, integrator=euler)
    ax1.plot(X, Y, '-o', label='Euler (Δx={})'.format(DX))

    Y = odeint(saturation_df, (0, Y0), X, integrator=heun)
    ax1.plot(X, Y, '-o', label='Heun (Δx={})'.format(DX))

    Y = odeint(saturation_df, (0, Y0), X)
    ax1.plot(X, Y, '-o', label='rk4 (Δx={})'.format(DX))

    X = np.arange(0, 1.25 * WIN[2], .01*DX)
    Y = odeint(saturation_df, (0, Y0), X)
    ax1.plot(X, Y, label='rk4 (Δx={})'.format(.01*DX))

    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y(x)')

    fig.show()

    # SAVE
    if args.save:
        fig.savefig('{}_heun_rk4.pdf'.format(args.save))

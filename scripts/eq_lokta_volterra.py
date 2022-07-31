import argparse
import numpy

import matplotlib.pyplot as plt

from scripts.ode.integrator import odeint


A = 2/3
B = 1
P = 4/3
Q = 1


def lv_df(x: float, Y: numpy.ndarray, a: float = A, b: float = B, p: float = P, q: float = Q):
    return numpy.array([(a-p*Y[1])*Y[0], (q*Y[0]-b)*Y[1]])


DX = 0.1
Y0 = 1
N = 10
WIN1 = (-.2, -.2, 2, 1.5)
X_MAX = 25
WIN2 = (0, 0, X_MAX, 2)
grid_increments = (0.1, 0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    fig = plt.figure(figsize=(9, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(WIN1[0], WIN1[2])
    ax1.set_ylim(WIN1[1], WIN1[3])

    ax1.set_xlabel('x (proies)')
    ax1.set_ylabel('y (prédateurs)')

    # stream
    x = numpy.arange(WIN1[0], WIN1[2], grid_increments[0])
    y = numpy.arange(WIN1[0], WIN1[2], grid_increments[1])
    X, Y = numpy.meshgrid(x, y)

    dx = numpy.zeros(X.shape)
    dy = numpy.zeros(X.shape)

    for i in range(len(x)):
        for j in range(len(y)):
            dx[j, i], dy[j, i] = lv_df(0, numpy.array([x[i], y[j]]).T)

    ax1.streamplot(X, Y, dx, dy, density=1, color='red', linewidth=.75)

    # a solution
    initial_conditions = [Y0, Y0]
    X = numpy.arange(0, X_MAX, DX)

    ax1.plot([B / Q], [A / P], 'bo')
    ax1.plot(initial_conditions[0], initial_conditions[1], 'bo')

    ax1.text(
        initial_conditions[0],
        initial_conditions[1],
        'x(0)={}, y(0)={}'.format(*initial_conditions),
        horizontalalignment='center',
        verticalalignment='bottom'
    )

    ax1.axhline(y=0, linestyle='--', color='grey')
    ax1.axvline(x=0, linestyle='--', color='grey')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim(WIN2[0], WIN2[2])
    ax2.set_ylim(WIN2[1], WIN2[3])
    ax2.set_xlabel('t')
    ax2.set_ylabel('Nombre')
    Y = odeint(
        lv_df,
        (0, numpy.array(initial_conditions).transpose()),
        X
    )

    ax1.plot(Y[:, 0], Y[:, 1], 'b')

    ax2.plot(X, Y[:, 0], label='x (proies)')
    ax2.plot(X, Y[:, 1], label='y (prédateurs)')

    ax2.legend()

    fig.show()

    # SAVE
    if args.save:
        fig.savefig('{}_lokta_volterra.pdf'.format(args.save))


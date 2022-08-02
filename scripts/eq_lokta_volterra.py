import argparse
import numpy

from scripts.ode.integrator import odeint
from scripts.ode.phase_portrait_2d import PhasePortrait2D


def create_lv_df(a: float, b: float, p: float, q: float):
    return lambda x, Y: numpy.array([(a-p*Y[1])*Y[0], (q*Y[0]-b)*Y[1]])


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

    # E.D.
    A = 2 / 3
    B = 1
    P = 4 / 3
    Q = 1
    lv_df = create_lv_df(A, B, P, Q)

    # stream plot
    fig = PhasePortrait2D(lv_df).create_figure(
        grid_increments=grid_increments,
        stream_density=1,
        graph_limits=WIN1,
        subplot_set=(1, 2, 1),
        graph_labels=('x (proies)', 'y (prédateurs)'),
        graph_size=(9, 5)
    )

    ax1 = fig.get_axes()[0]

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


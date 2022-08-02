import argparse
import numpy

from scripts.ode.phase_portrait_2d import PhasePortrait2D


def create_lv_competitive_df(a: float, b: float, c: float, d: float, p: float, q: float):
    return lambda x, Y: numpy.array([(a-b*Y[0]-p*Y[1])*Y[0], (c-d*Y[1]-q*Y[0])*Y[1]])


DX = 0.1
Y0 = 1
N = 10
WIN1 = (-1, -1, 35, 35)
grid_increments = (0.1, 0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # extinct
    A, B, C, D, P, Q = 24, 1, 30., 1, 2, 2
    lv_extinct = create_lv_competitive_df(A, B, C, D, P, Q)

    fig_extinct = PhasePortrait2D(lv_extinct).create_figure(
        grid_increments=grid_increments,
        stream_density=1.25,
        graph_limits=WIN1,
        graph_labels=('x (espèce 1)', 'y (espèce 2)'),
        graph_size=(4.5, 5)
    )

    criticals_extinct = [0, A / B, 0, (A * D - P * C) / (B * D - Q * P)], [0, 0, C / D, (B * C - Q * A) / (B * D - Q * P)]

    ax1 = fig_extinct.get_axes()[0]
    ax1.plot(criticals_extinct[0], criticals_extinct[1], 'o')

    for x, y in zip(*criticals_extinct):
        ax1.text(x, y, '({}, {})'.format(x, y))

    fig_extinct.show()

    # inhib
    A, B, C, D, P, Q = 24, 1, 30., 1, 1/2, 1/2
    lv_inhib = create_lv_competitive_df(A, B, C, D, P, Q)

    fig_inhib = PhasePortrait2D(lv_inhib).create_figure(
        grid_increments=grid_increments,
        stream_density=1.25,
        graph_limits=WIN1,
        graph_labels=('x (espèce 1)', 'y (espèce 2)'),
        graph_size=(4.5, 5)
    )

    criticals_inhib = [0, A / B, 0, (A * D - P * C) / (B * D - Q * P)], [0, 0, C / D, (B * C - Q * A) / (B * D - Q * P)]

    ax1 = fig_inhib.get_axes()[0]
    ax1.plot(criticals_inhib[0], criticals_inhib[1], 'o')

    for x, y in zip(*criticals_inhib):
        ax1.text(x, y, '({}, {})'.format(x, y))

    fig_inhib.show()

    # SAVE
    if args.save:
        fig_extinct.savefig('{}_lv_competitive_e.pdf'.format(args.save))
        fig_inhib.savefig('{}_lv_competitive_i.pdf'.format(args.save))


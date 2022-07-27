import argparse

import numpy

from scripts.ode.phase_portrait_2d import PhasePortrait2D

MARGIN_LEFT = .2
MARGIN_BOTTOM = .15

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    fig_nn = PhasePortrait2D(numpy.array([[-2, 1], [1, -2]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_nn.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_pp = PhasePortrait2D(numpy.array([[2, 1], [1, 2]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_pp.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_n = PhasePortrait2D(numpy.array([[-7, 1], [-4, -3]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_n.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_p = PhasePortrait2D(numpy.array([[7, 1], [-4, 3]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_p.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_np = PhasePortrait2D(numpy.array([[1, 2], [3, 2]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_np.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_ip = PhasePortrait2D(numpy.array([[3, -13], [5, 1]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_ip.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_ip.show()

    fig_in = PhasePortrait2D(numpy.array([[-3, -13], [5, -1]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_in.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_in.show()

    fig_o = PhasePortrait2D(numpy.array([[3, 9], [-4, -3]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_o.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_o.show()

    if args.save:
        fig_nn.savefig('{}_phase2d_nn.pdf'.format(args.save))
        fig_pp.savefig('{}_phase2d_pp.pdf'.format(args.save))
        fig_n.savefig('{}_phase2d_n.pdf'.format(args.save))
        fig_p.savefig('{}_phase2d_p.pdf'.format(args.save))
        fig_np.savefig('{}_phase2d_np.pdf'.format(args.save))
        fig_in.savefig('{}_phase2d_in.pdf'.format(args.save))
        fig_ip.savefig('{}_phase2d_ip.pdf'.format(args.save))
        fig_o.savefig('{}_phase2d_o.pdf'.format(args.save))

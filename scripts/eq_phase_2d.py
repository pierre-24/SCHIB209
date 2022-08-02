import argparse

import numpy

from scripts.ode.phase_portrait_2d import PhasePortraitAutonomous2D

MARGIN_LEFT = .2
MARGIN_BOTTOM = .15

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    fig_nn = PhasePortraitAutonomous2D(numpy.array([[-2, 1], [1, -2]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_nn.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_pp = PhasePortraitAutonomous2D(numpy.array([[2, 1], [1, 2]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_pp.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_n = PhasePortraitAutonomous2D(numpy.array([[-7, 1], [-4, -3]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_n.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_p = PhasePortraitAutonomous2D(numpy.array([[7, 1], [-4, 3]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_p.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_np = PhasePortraitAutonomous2D(numpy.array([[1, 2], [3, 2]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_np.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_ip = PhasePortraitAutonomous2D(numpy.array([[3, -13], [5, 1]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_ip.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_in = PhasePortraitAutonomous2D(numpy.array([[-3, -13], [5, -1]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_in.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_o = PhasePortraitAutonomous2D(numpy.array([[3, 9], [-4, -3]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_o.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_in.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_n0 = PhasePortraitAutonomous2D(numpy.array([[-1, 2], [1, -2]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_n0.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_n0.show()

    fig_p0 = PhasePortraitAutonomous2D(numpy.array([[1, 2], [1, 2]])).create_figure(
        graph_limits=(-2, -2, 2, 2),
        graph_size=(3.5, 3.5)
    )

    fig_p0.subplots_adjust(left=MARGIN_LEFT, bottom=MARGIN_BOTTOM)

    fig_p0.show()

    if args.save:
        fig_nn.savefig('{}_phase2d_nn.pdf'.format(args.save))
        fig_pp.savefig('{}_phase2d_pp.pdf'.format(args.save))
        fig_n.savefig('{}_phase2d_n.pdf'.format(args.save))
        fig_p.savefig('{}_phase2d_p.pdf'.format(args.save))
        fig_np.savefig('{}_phase2d_np.pdf'.format(args.save))
        fig_in.savefig('{}_phase2d_in.pdf'.format(args.save))
        fig_ip.savefig('{}_phase2d_ip.pdf'.format(args.save))
        fig_o.savefig('{}_phase2d_o.pdf'.format(args.save))
        fig_n0.savefig('{}_phase2d_n0.pdf'.format(args.save))
        fig_p0.savefig('{}_phase2d_p0.pdf'.format(args.save))

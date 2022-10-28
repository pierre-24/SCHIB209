import argparse

import numpy

from scripts.fourier.series import FourierSeriesPlot


WIN = (-10, -5, 10, 5)
L = 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # PLOT
    plot = FourierSeriesPlot(
        L=L,
        f=lambda x: -x,
        b_k=lambda k: 6*(-1)**k / (k * numpy.pi)
    )

    fig = plot.create_figure(
        N=list(5*i for i in range(5)),
        graph_limits=(-9, -5, 9, 5),
        sol_label='y(x) sur [-{0},{0}]'.format(L)
    )
    fig.show()

    figb = plot.create_details_figure(10)
    figb.show()

    # SAVE
    if args.save:
        fig.savefig('{}_fourier.pdf'.format(args.save))
        figb.savefig('{}_details_fourier.pdf'.format(args.save))

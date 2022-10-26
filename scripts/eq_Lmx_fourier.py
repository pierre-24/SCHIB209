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
    fig = FourierSeriesPlot(
        L=L,
        f=lambda x: -x,
        b_k=lambda k: 6*(-1)**k / (k * numpy.pi)
    ).create_figure(
        N=list(5*i for i in range(5)),
        graph_limits=(-9, -5, 9, 5),
        sol_label='y(x) sur [-{0},{0}]'.format(L)
    )
    fig.show()

    # SAVE
    if args.save:
        fig.savefig('{}_fourier.pdf'.format(args.save))

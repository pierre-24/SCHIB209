import argparse

import numpy
import numpy as np

from scripts.fourier.series import FourierSeriesPlot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # PLOT
    plot = FourierSeriesPlot(
        L=numpy.pi,
        f=lambda x: x**2,
        a_0=np.pi**2 / 3,
        a_k=lambda k: 4 * (-1)**k / k**2
    )

    fig = plot.create_figure(
        N=list(2*i for i in range(6)),
        graph_limits=(-10, -1, 10, 10),
        sol_label='y(x) sur [-π,π]'
    )
    fig.show()

    figb = plot.create_details_figure(5)
    figb.show()

    # SAVE
    if args.save:
        fig.savefig('{}_fourier.pdf'.format(args.save))
        figb.savefig('{}_details_fourier.pdf'.format(args.save))

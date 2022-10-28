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
        f=lambda x: numpy.where(x < 0, 0, x),
        a_0=np.pi / 4,
        a_k=lambda k: ((-1)**k - 1) / (k**2 * numpy.pi),
        b_k=lambda k: (-1)**(k + 1) / k
    )

    fig = plot.create_figure(
        N=list(5*i for i in range(5)),
        graph_limits=(-10, -.5, 10, 5),
        sol_label='y(x) sur [-π,π]'
    )
    fig.show()

    figb = plot.create_details_figure(10)
    figb.show()

    # SAVE
    if args.save:
        fig.savefig('{}_fourier.pdf'.format(args.save))
        figb.savefig('{}_details_fourier.pdf'.format(args.save))

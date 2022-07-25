import argparse
import numpy

from scripts.ode.directional_1d import Directional1D
from scripts.ode.phase_portrait_autonomous_1d import PhasePortraitAutonomous1D


def sin_df(x: float, y: float) -> float:
    """
    Sinus-like equation `:math:`\\frac{dy}{dx} = y * sin(y/\\pi)`.
    """
    return y * numpy.sin(y * numpy.pi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # directional field
    fd = Directional1D(sin_df).create_figure(
        initial_values=[(0, -.5), (0, .5), (0, 1.75), (0, 3.25)],
        graph_limits=(-.25, -2, 4, 4),
        grid_increments=(.25, .25),
        graph_labels=('t', 'ω(t)'),
        graph_size=(4, 4)
    )

    # phase portrait
    fp = PhasePortraitAutonomous1D(lambda N: sin_df(0, N)).create_figure(
        graph_limits=(-4.5, -5, 4.5, 5),
        graph_labels=('ω', 'f(ω)'),
        zeros_estimate=list(range(-4, 5)),
        arrow_size=.1,
        arrow_sep=.2,
        graph_size=(4, 4)
    )

    fd.show()
    fp.show()

    if args.save:
        fd.savefig('{}_directional.pdf'.format(args.save))
        fp.savefig('{}_phase_portrait.pdf'.format(args.save))


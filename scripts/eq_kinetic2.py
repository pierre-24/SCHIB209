import argparse
import numpy

from scripts.ode.directional_1d import Directional1D
from scripts.ode.phase_portrait_autonomous_1d import PhasePortraitAutonomous1D


def kin2_df(t: float, x: float, k: float = 1, a: float = 1, b: float = 2) -> float:
    """
    Order 2 kinetic `:math:`\\frac{dx}{dt} = k\\,(a-x)(b-x)`.
    """
    return k * (a - x) * (b - x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # directional field
    fd = Directional1D(kin2_df).create_figure(
        initial_values=[(0, 0), (0, .5), (0, 1.75), (0, 2.25)],
        graph_limits=(-.25, -1, 4, 3),
        grid_increments=(.25, .25),
        graph_labels=('t', 'x(t)'),
        graph_size=(4, 4)
    )

    # phase portrait
    fp = PhasePortraitAutonomous1D(lambda N: kin2_df(0, N)).create_figure(
        graph_limits=(0, -1, 3, 1),
        graph_labels=('x', 'f(x)'),
        zeros_estimate=[1, 2],
        arrow_size=.05,
        arrow_sep=.1,
        graph_size=(4, 4)
    )

    if not args.save:
        fd.show()
        fp.show()
    else:
        fd.savefig('{}_directional.pdf'.format(args.save))
        fp.savefig('{}_phase_portrait.pdf'.format(args.save))


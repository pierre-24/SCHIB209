import argparse

from scripts.ode.directional_1d import Directional1D
from scripts.ode.phase_portrait_autonomous_1d import PhasePortraitAutonomous1D


def logistic_df(t: float, N: float, r: float = 1, Nf: float = 1) -> float:
    """
    Logistic equation `:math:`\\frac{dN}{dt} = r\,N\,(N_\\infty - N)`.
    """
    return r * N * (Nf - N)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    # directional field
    fd = Directional1D(logistic_df).create_figure(
        initial_values=[(0, -.25), (0, .25), (0, 1), (0, 3)],
        graph_limits=(-.25, -1, 4, 2),
        grid_increments=(.2, .2),
        graph_labels=('t', 'N(t)')
    )

    # phase portrait
    fp = PhasePortraitAutonomous1D(lambda N: logistic_df(0, N)).create_figure(
        graph_limits=(-1, -1, 2, 0.5),
        graph_labels=('N', 'f(N)'),
        zeros_estimate=[0, 1]
    )

    if not args.save:
        fd.show()
        fp.show()
    else:
        fd.savefig('{}_directional.pdf'.format(args.save))
        fp.savefig('{}_phase_portrait.pdf'.format(args.save))


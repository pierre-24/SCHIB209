from typing import List, Tuple, Callable
import numpy


def rk4(df, x: float, y: float, dx: float) -> List[float]:
    """Runge-Kutta integrator 4 (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
    """

    # step 1
    k = df(x, y)
    ys = y + dx / 6 * k

    # step 2
    k = df(x + dx / 2, y + dx / 2 * k)
    ys += dx / 3 * k

    # step 3
    k = df(x + dx / 2, y + dx / 2 * k)
    ys += dx / 3 * k

    # step 4
    k = df(x + dx, y + dx * k)
    ys += dx / 6 * k

    return ys


def odeint(
        df: Callable[[float, float], float],
        initial_conditions: Tuple[float, float],
        X: numpy.ndarray) -> numpy.ndarray:
    """Compute the solution of a 1D ODE for different points X"""

    Y = numpy.zeros(X.shape[0])

    X[0], Y[0] = initial_conditions

    for i in range(1, X.shape[0]):
        Y[i] = rk4(df, X[i - 1], Y[i - 1], X[i] - X[i - 1])

    return Y
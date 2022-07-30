from typing import List, Tuple, Callable, Union
import numpy


type_f_or_arr = Union[float, numpy.ndarray]
type_df = Union[Callable[[float, float], float], Callable[[float, numpy.ndarray], numpy.ndarray]]


def euler(df: type_df, x0: float, y0: type_f_or_arr, dx: float) -> type_f_or_arr:
    """Euler integrator
    """
    return y0 + df(x0, y0) * dx


def heun(df: type_df, x0: float, y0: type_f_or_arr, dx: float) -> type_f_or_arr:
    """Heun integrator (https://en.wikipedia.org/wiki/Heun%27s_method)"""

    k1 = df(x0, y0)
    k2 = df(x0 + dx, y0 + k1 * dx)

    return y0 + .5 * (k1 + k2) * dx


def rk4(df: type_df, x0: float, y0: type_f_or_arr, dx: float) -> type_f_or_arr:
    """Runge-Kutta integrator 4 (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
    """

    # step 1
    k = df(x0, y0)
    ys = y0 + dx / 6 * k

    # step 2
    k = df(x0 + dx / 2, y0 + dx / 2 * k)
    ys += dx / 3 * k

    # step 3
    k = df(x0 + dx / 2, y0 + dx / 2 * k)
    ys += dx / 3 * k

    # step 4
    k = df(x0 + dx, y0 + dx * k)
    ys += dx / 6 * k

    return ys


def odeint(
        df: type_df,
        initial_conditions: Tuple[float, type_f_or_arr],
        X: numpy.ndarray,
        integrator: Callable[[type_df, float, type_f_or_arr, float], type_f_or_arr] = rk4
) -> numpy.ndarray:
    """Compute the solution of a 1D ODE for different points X"""

    if issubclass(type(initial_conditions[1]), numpy.ndarray):
        Y = numpy.zeros((X.shape[0], initial_conditions[1].shape[0]))
    else:
        Y = numpy.zeros(X.shape[0])

    X[0] = initial_conditions[0]
    Y[0] = initial_conditions[1]

    for i in range(1, X.shape[0]):
        Y[i] = integrator(df, X[i - 1], Y[i - 1], X[i] - X[i - 1])

    return Y

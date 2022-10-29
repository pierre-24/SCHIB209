import argparse

import numpy
from scipy import fft

import matplotlib.pyplot as plt

PI = numpy.pi


dx = 1e-2
MX = 100

MXA = 3
MXE = 5


def f(T: numpy.ndarray):
    return numpy.where(numpy.abs(T) < .25, 1, 0)
    #return numpy.where(T > 0, numpy.exp(-.5*T), 0) * numpy.sin(2*2*PI*T)
    #return numpy.sin(2*2*PI*T) + .5 * numpy.sin(9*2*PI*T) + .25 * numpy.sin(4*2*PI*T) + numpy.random.random(T.shape[0]) * 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='save instead of show', type=str)

    args = parser.parse_args()

    T = numpy.arange(-MX, MX, dx)
    Yt = f(T)
    W = fft.fftshift(fft.fftfreq(int(2 * MX / dx), dx))
    Yw = fft.fftshift(fft.fft(Yt))

    fig = plt.figure(figsize=(6, 10))

    ax = fig.add_subplot(3, 1, 1)
    ax.set_xlim(-MXA, MXA)
    ax.set_xlabel('t')
    ax.set_ylabel('y(t)')
    ax.plot(T, Yt)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_xlabel('w')
    ax2.set_ylabel('A(ω)')
    ax2.set_xlim(-MXE, MXE)
    ax2.plot(W, numpy.abs(Yw), label='Amplitude')

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_xlabel('w')
    ax3.set_ylabel('Φ(ω)')
    ax3.set_ylim(-numpy.pi, numpy.pi)
    ax3.set_xlim(0, MXE)
    ax3.plot(W, numpy.arctan2(
        Yw.imag,
        Yw.real
    ))

    print(Yw.imag.max(), Yw.imag.min())

    fig.show()

    # SAVE
    if args.save:
        fig.savefig('{}_tf.pdf'.format(args.save))

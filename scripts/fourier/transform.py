import numpy
from scipy import fft


def image_2dft(im: numpy.ndarray) -> numpy.ndarray:
    im = fft.ifftshift(im)
    ft = fft.fft2(im)
    return fft.fftshift(ft)


def image_i2dft(ft: numpy.ndarray) -> numpy.ndarray:
    ft = fft.fftshift(ft)
    im = fft.ifft2(ft)
    return fft.ifftshift(im)

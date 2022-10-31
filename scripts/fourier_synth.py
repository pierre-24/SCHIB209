import argparse
from PIL import Image
import numpy

from scripts.fourier.transform import image_2dft, image_i2dft


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input image', type=str)

    args = parser.parse_args()

    image = numpy.asarray(Image.open(args.input).convert('L'))
    N = '.'.join(args.input.split('.')[:-1])

    ft = image_2dft(image)
    ftm = numpy.abs(ft)
    ftm = 255 - ftm / (.4 * ftm.max()) * 255
    im_rec = image_i2dft(numpy.abs(ft))

    ftp = (numpy.arctan2(ft.imag, ft.real) + numpy.pi) / (2 * numpy.pi) * 255

    Image.fromarray(ftm).convert('L').save('{}_ft.png'.format(N))
    Image.fromarray(ftp).convert('L').save('{}_ftp.png'.format(N))
    Image.fromarray(im_rec.real).convert('L').save('{}_ift.png'.format(N))





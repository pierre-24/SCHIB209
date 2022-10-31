import argparse
from PIL import Image
import numpy

from scripts.fourier.transform import image_2dft, image_i2dft


FROM_IM = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', required=True, help='save image', type=str)

    args = parser.parse_args()

    if FROM_IM:
        image = numpy.zeros((0, 0))
        ft = image_2dft(image)
    else:
        ft = numpy.zeros((100, 100))
        A = 10
        ft[50 - A, 50 - A] = ft[50 + A, 50 + A] = 2000000

    ftm = numpy.abs(ft)
    ftm = 255 - ftm / ftm.max() * 255

    Image.fromarray(ftm).convert('L').save('{}_ft.png'.format(args.save))

    im_rec = image_i2dft(numpy.abs(ft))
    print(im_rec.max())
    Image.fromarray(im_rec.real).convert('L').save('{}_ift.png'.format(args.save))





import argparse
from PIL import Image, ImageDraw
import numpy

from typing import Callable

from scripts.fourier.transform import image_2dft

SZ = 257


def gen_im(name: str, n: int, func_draw: Callable[[ImageDraw], None], mag: float = 0.8):
    im = Image.new('L', (SZ, SZ), (255, ))
    draw = ImageDraw.Draw(im)

    func_draw(draw)

    im.save('{}_G{}.png'.format(name, n))

    image = 255 - numpy.asarray(im.convert('L'))
    ft = image_2dft(image)
    ftm = numpy.abs(ft)

    ftm[1:] += ftm[:SZ-1]
    ftm[:SZ-1] += ftm[1:]
    ftm[:, 1:] += ftm[:, :SZ-1]
    ftm[:, :SZ-1] += ftm[:, 1:]

    ftm = 255 - ftm / (mag * ftm.max()) * 255

    Image.fromarray(ftm).convert('L').save('{}_P{}.png'.format(name, n))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', required=True, help='save image', type=str)

    args = parser.parse_args()

    # G1 - horizontal slits
    def draw_g1(draw: ImageDraw):
        INC = (22, 8)
        N = (12, 32)

        for y in range(N[1]):
            for x in range(N[0]):
                draw.line((INC[0] * x + 4, INC[1] * y + 4, INC[0] * x + 12, INC[1] * y + 4), width=1)

    gen_im(args.save, 1, draw_g1)

    # G2 - tilted slits (zig)
    def draw_g2(draw: ImageDraw):
        INC = (22, 8)
        N = (12, 32)

        for y in range(N[1]):
            for x in range(N[0]):
                draw.line((INC[0] * x + 4, INC[1] * y + 3, INC[0] * x + 12, INC[1] * y + 9), width=1)

    gen_im(args.save, 2, draw_g2)

    # G4 - zigzag
    def draw_g4(draw: ImageDraw):
        INC = (16, 8)
        N = (16, 33)

        for x in range(N[0]):
            draw.line([
                (INC[0] * x + (0 if y % 2 == 0 else 8), INC[1] * y - 0) for y in range(N[1])
            ], width=1, joint='curve')

    gen_im(args.save, 4, draw_g4)

    # G5 - zigzag squized
    def draw_g5(draw: ImageDraw):
        INC = (16, 8)
        N = (16, 33)

        for x in range(N[0]):
            draw.line([
                (INC[0] * x + (0 if y % 2 == 0 else 14), INC[1] * y - 0) for y in range(N[1])
            ], width=2)

    gen_im(args.save, 5, draw_g5)

    # G6 - true sin
    def draw_g6(draw: ImageDraw):
        INC = (16,)
        N = (16,)

        for x in range(N[0]):
            Y = numpy.arange(0, 257)
            X = (4 + x * INC[0] + 7 * numpy.sin(Y * 2 * numpy.pi / 40 + 2)).astype(int)
            draw.line(list(zip(X, Y)), width=1)

    gen_im(args.save, 6, draw_g6)

    # G7 - true double sin
    def draw_g7(draw: ImageDraw):
        INC = (16,)
        N = (16,)

        for x in range(N[0]):
            Y = numpy.arange(0, 257)
            X = (4 + x * INC[0] + 7 * numpy.sin(Y * 2 * numpy.pi / 40 + 2)).astype(int)
            X2 = (4 + x * INC[0] + 7 * numpy.sin(Y * 2 * numpy.pi / 40 + 6)).astype(int)
            draw.line(list(zip(X, Y)))
            draw.line(list(zip(X2, Y)))

    gen_im(args.save, 7, draw_g7)

    # G8 - dot
    def draw_g8(draw: ImageDraw):
        INC = (16,)
        N = (16,)

        for x in range(N[0]):
            Y = 4 * numpy.arange(0, 129)
            X = (4 + x * INC[0] + 8 * numpy.sin(Y * 2 * numpy.pi / 40 + 2)).astype(int)
            draw.point(list(zip(X, Y)))

            for xy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                draw.point(list(zip(X + xy[0], Y + xy[1])), fill=150)

            for xy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                draw.point(list(zip(X + xy[0], Y + xy[1])), fill=200)

    gen_im(args.save, 8, draw_g8)

    # G9 - double dot
    def draw_g9(draw: ImageDraw):
        INC = (16,)
        N = (16,)

        for x in range(N[0]):
            Y = 4 * numpy.arange(0, 65)
            X = (4 + x * INC[0] + 7 * numpy.sin(Y * 2 * numpy.pi / 40 + 2)).astype(int)
            X2 = (4 + x * INC[0] + 7 * numpy.sin(Y * 2 * numpy.pi / 40 + 6)).astype(int)

            for i in range(Y.shape[0]):
                x1, x2 = X[i], X2[i]
                if x1 > x2:
                    x1, x2 = x2, x1
                draw.line([(x1, Y[i]), (x2, Y[i])], fill=150)

            draw.line(list(zip(X, Y)), fill=150)
            draw.line(list(zip(X2, Y)), fill=150)

            for xy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                draw.point(list(zip(X + xy[0], Y + xy[1])), fill=150)
                draw.point(list(zip(X2 + xy[0], Y + xy[1])), fill=150)

            for xy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                draw.point(list(zip(X + xy[0], Y + xy[1])), fill=200)
                draw.point(list(zip(X2 + xy[0], Y + xy[1])), fill=200)

            draw.point(list(zip(X, Y)))
            draw.point(list(zip(X2, Y)))

    gen_im(args.save, 9, draw_g9)


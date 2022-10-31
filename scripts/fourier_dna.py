import argparse
from PIL import Image, ImageDraw
import numpy

from typing import Callable

from scripts.fourier.transform import image_2dft

SZ = 257
P = 40
A = 7
T1 = -20
T2 = -16


def gen_im(name: str, n: int, func_draw: Callable[[ImageDraw], None], mag: float = 1):
    im = Image.new('L', (SZ, SZ), (255, ))
    draw = ImageDraw.Draw(im)

    func_draw(draw)

    im.save('{}_G{}.png'.format(name, n))

    image = 255 - numpy.asarray(im.convert('L'))
    ft = image_2dft(image)
    ftm = numpy.abs(ft)

    mx = ftm.max()

    ftm[1:] += ftm[:SZ-1]
    ftm[:SZ-1] += ftm[1:]
    ftm[:, 1:] += ftm[:, :SZ-1]
    ftm[:, :SZ-1] += ftm[:, 1:]

    ftm = 255 - ftm / (mag * mx) * 255

    Image.fromarray(ftm).convert('L').save('{}_P{}.png'.format(name, n))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', required=True, help='save image', type=str)

    args = parser.parse_args()

    # G1 - horizontal slits
    def draw_g1(draw: ImageDraw):
        INC = (22, 16)
        N = (12, 16)

        for y in range(N[1]):
            for x in range(N[0]):
                draw.line((INC[0] * x + 0, INC[1] * y + 4, INC[0] * x + 14, INC[1] * y + 4), width=1)

    gen_im(args.save, 1, draw_g1, mag=0.9)

    # G2 - tilted slits (zig)
    def draw_g2(draw: ImageDraw):
        INC = (16, 40)
        N = (17, 7)

        for y in range(N[1]):
            for x in range(N[0]):
                draw.line((INC[0] * x + 0, INC[1] * y + 0, INC[0] * x + 14, INC[1] * y + P / 2), width=1)

    gen_im(args.save, 2, draw_g2, mag=0.9)

    # G4 - zigzag
    def draw_g4(draw: ImageDraw):
        INC = (16, P / 2)
        N = (16, 15)

        for x in range(N[0]):
            draw.line([
                (INC[0] * x + (0 if y % 2 == 0 else 2 * A), INC[1] * y - 0) for y in range(N[1])
            ], width=1)

    gen_im(args.save, 4, draw_g4)

    # G5 - zigzag squized
    def draw_g5(draw: ImageDraw):
        INC = (16, 12)
        N = (16, 23)

        for x in range(N[0]):
            draw.line([
                (INC[0] * x + (0 if y % 2 == 0 else 2 * A), INC[1] * y - 0) for y in range(N[1])
            ], width=1)

    gen_im(args.save, 5, draw_g5)

    # G6 - true sin
    Y = numpy.arange(0, 257)
    SX = A * numpy.sin(Y * 2 * numpy.pi / P + T1)
    SX2 = A * numpy.sin(Y * 2 * numpy.pi / P + T2)

    def draw_g6(draw: ImageDraw):
        INC = (16,)
        N = (17,)

        for x in range(N[0]):
            X = (A + x * INC[0] + SX).astype(int)
            draw.line(list(zip(X, Y)), width=1)

    gen_im(args.save, 6, draw_g6)

    # G7 - true double sin
    def draw_g7(draw: ImageDraw):
        INC = (16,)
        N = (17,)

        for x in range(N[0]):
            X = (x * INC[0] + SX).astype(int)
            X2 = (x * INC[0] + SX2).astype(int)
            draw.line(list(zip(X, Y)))
            draw.line(list(zip(X2, Y)))

    gen_im(args.save, 7, draw_g7)

    # G8 - dot
    Y = 4 * numpy.arange(0, 129)
    SX = A * numpy.sin(Y * 2 * numpy.pi / P + T1)
    SX2 = A * numpy.sin(Y * 2 * numpy.pi / P + T2)

    def draw_g8(draw: ImageDraw):
        INC = (16,)
        N = (17,)

        for x in range(N[0]):
            X = (A + x * INC[0] + SX).astype(int)

            draw.point(list(zip(X, Y)))

            # enhance dot
            for xy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                draw.point(list(zip(X + xy[0], Y + xy[1])), fill=150)

            for xy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                draw.point(list(zip(X + xy[0], Y + xy[1])), fill=200)

    gen_im(args.save, 8, draw_g8)

    # G9 - double dot
    def draw_g9(draw: ImageDraw):
        INC = (16,)
        N = (17,)

        for x in range(N[0]):
            X = (A + x * INC[0] + SX).astype(int)
            X2 = (A + x * INC[0] + SX2).astype(int)

            for xy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                draw.point(list(zip(X + xy[0], Y + xy[1])), fill=150)
                draw.point(list(zip(X2 + xy[0], Y + xy[1])), fill=150)

            for xy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                draw.point(list(zip(X + xy[0], Y + xy[1])), fill=200)
                draw.point(list(zip(X2 + xy[0], Y + xy[1])), fill=200)

            draw.point(list(zip(X, Y)))
            draw.point(list(zip(X2, Y)))

    gen_im(args.save, 9, draw_g9)

    # G10 - double dot + helix
    def draw_g10(draw: ImageDraw):
        INC = (16,)
        N = (17,)

        for x in range(N[0]):
            X = (A + x * INC[0] + SX).astype(int)
            X2 = (A + x * INC[0] + SX2).astype(int)

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


    gen_im(args.save, 10, draw_g10)


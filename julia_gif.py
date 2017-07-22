import tensorflow as tf
import numpy as np
from PIL import Image
import os
from moviepy.editor import ImageSequenceClip


def gif(filename, array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

R = 4
ITER_NUM = 200


def get_color(bg_ratio, ratio):
    def color(z, i):
        if abs(z) < R:
            return 0, 0, 0
        v = np.log2(i + R - np.log2(np.log2(abs(z)))) / 5
        if v < 1.0:
            return v**bg_ratio[0], v**bg_ratio[1], v ** bg_ratio[2]
        else:
            v = max(0, 2 - v)
            return v**ratio[0], v**ratio[1], v**ratio[2]
    return color


def gen_julia(Z, c, bg_ratio, ratio):
    xs = tf.constant(np.full(shape=Z.shape, fill_value=c, dtype=Z.dtype))
    zs = tf.Variable(Z)
    ns = tf.Variable(tf.zeros_like(xs, tf.float32))
    with tf.Session():
        tf.global_variables_initializer().run()
        zs_ = tf.where(tf.abs(zs) < R, zs**2 + xs, zs)
        not_diverged = tf.abs(zs_) < R
        step = tf.group(
            zs.assign(zs_),
            ns.assign_add(tf.cast(not_diverged, tf.float32))
        )

        for i in range(ITER_NUM):
            step.run()
        final_step = ns.eval()
        final_z = zs_.eval()
    r, g, b = np.frompyfunc(get_color(bg_ratio, ratio), 2, 3)(final_z, final_step)
    img_array = np.dstack((r, g, b))
    return Image.fromarray(np.uint8(img_array * 255))


if __name__ == '__main__':
    n = 60
    start_x = -1.9  # x range
    end_x = 1.9
    start_y = -1.1  # y range
    end_y = 1.1
    width = 600  # image width
    bg_ratio = (4, 2.5, 1)
    ratio = (0.9, 0.9, 0.9)
    step = (end_x - start_x) / width
    Y, X = np.mgrid[start_y:end_y:step, start_x:end_x:step]
    Z = X + 1j * Y

    seqs = np.zeros([n] + list(Z.shape) + [3])
    for i in range(0, n):
        print('Generating {}/{}....'.format(i + 1, n))
        theta = 2 * np.pi / n * i
        c = -(0.835 - 0.1 * np.cos(theta)) - (0.2321 + 0.1 * np.sin(theta)) * 1j
        img = gen_julia(Z, c, bg_ratio, ratio)
        seqs[i, :, :] = np.array(img)
    print('Make gif.....')
    gif('julia_gif.gif', seqs, 8)
    print('Please check julia_gif.gif...')

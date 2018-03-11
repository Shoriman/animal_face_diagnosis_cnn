import numpy as np

from chainer.dataset import convert


def mean_subst(data, mean_img):
    for d in data:
        np.substract(d[0], mean_img, out=d[0])
    return data


def calc_mean_img(data):
    images = convert.concat_examples(data)[0]
    mean_img = np.mean(images, axis=0)
    return mean_img

import argparse

import numpy as np
from chainer import serializers

from process_dataset.proc_dataset import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('data')
parser.add_argument('model')
parser.add_argument('--norm', '-n', type=int, default=0,
                    help='Input-normalization mode')
parser.add_argument('--mean_img', default=None)
args = parser.parse_args()

mean_img = np.load(args.mean_img)

test_info = get_dataset(args.data, norm=args.norm, mean_img=mean_img)
test = test_info.__getitem__(slice(0, test_info.__len__(), 1))

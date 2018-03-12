import argparse
import json
import os
import collections as cl

import numpy as np
from chainer import serializers

from process_dataset.proc_dataset import get_dataset
from models.discriminator import discriminator
from evaluation.evaluator import evaluator

parser = argparse.ArgumentParser(description='Test discriminator')
parser.add_argument('data')
parser.add_argument('model')
parser.add_argument('arch_path', help='Model architecture path')
parser.add_argument('param_path', help='Parameter path')
parser.add_argument('out_dir', help='Output directory path')
parser.add_argument('--norm', '-n', type=int, default=0,
                    help='Input-normalization mode')
parser.add_argument('--mean_img', default=None)
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID')
parser.add_argument('--bsize', '-b', type=int, default=10,
                    help='Batch size for test')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

test_dt_names = ['data', 'model', 'arch', 'param', 'norm',
                 'mean_img', 'gpu', 'bsize']
test_dt_lst = [args.data, args.model, args.arch_path, args.param_path,
               args.norm, args.mean_img, args.gpu, args.bsize]
test_dt_dic = cl.OrderedDict()
for n, c in zip(test_dt_names, test_dt_lst):
    test_dt_dic[n] = c
with open(os.path.join(args.out_dir, 'test_detail.json'), 'w') as fp:
    json.dump(test_dt_dic, fp)

if args.mean_img is None:
    mean_img = None
else:
    mean_img = np.load(args.mean_img)
test_info = get_dataset(args.data, norm=args.norm, mean_img=mean_img)
test = test_info.__getitem__(slice(0, test_info.__len__(), 1))

with open(args.param_path) as fp:
    param = json.load(fp)
with open(args.arch_path) as fp:
    config = json.load(fp)

predictor = discriminator(config, dr=param['dr'], bn=param['bn'])
serializers.load_npz(args.model, predictor)

e = evaluator(test, predictor, bsize=args.bsize)
results = e.evaluate(device=args.gpu)

with open(os.path.join(args.out_dir, 'results.json'), 'w') as fp:
    json.dump(results, fp)

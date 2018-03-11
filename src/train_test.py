import os
import argparse
import collections as cl
import json

import numpy as np

from models.discriminator import discriminator
from models.custom_classifier import custom_classifier
from optimization.trainer import train_loop
from process_dataset.proc_dataset import get_dataset

parser = argparse.ArgumentParser(description='Train discriminator')
parser.add_argument('train_path', help='Train data path')
parser.add_argument('test_path', help='Test data path')
parser.add_argument('out_dir', help='Output directory path')
parser.add_argument('arch_path', help='Model architecture path')
parser.add_argument('param_path', help='Parameter path')
parser.add_argument('--norm', '-n', type=int, default=0,
                    help='Input-normalization mode')
parser.add_argument('--optname', '-o', default='MomentumSGD',
                    help='Optimizer [SGD, MomentumSGD, Adam]')
parser.add_argument('--epoch', '-e', type=int, default=50, help='Epoch Number')
parser.add_argument('--test_bsize', '-b', type=int, default=10,
                    help='Batch size for test')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID')
parser.add_argument('--mname', '-m', default='model', help='Saved model name')
parser.add_argument('--lr_attr', '-l', default='lr',
                    help='Learning rate attribute')
parser.add_argument('--mean_img', default=None)
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

train_dt_names = ['train', 'test', 'param', 'config', 'norm', 'optimizer',
                  'epoch', 'test_bsize', 'gpu']
train_dt_lst = [args.train_path, args.test_path, args.param_path,
                args.arch_path, args.norm, args.optname, args.epoch,
                args.test_bsize, args.gpu]
train_dt_dic = cl.OrderedDict()
for n, c in zip(train_dt_names, train_dt_lst):
    train_dt_dic[n] = c
with open(os.path.join(args.out_dir, 'train_detail.json'), 'w') as fp:
    json.dump(train_dt_dic, fp)

if args.mean_img is None:
    mean_img = None
else:
    mean_img = np.load(args.mean_img)
train_info = get_dataset(args.train_path, norm=args.norm, mean_img=mean_img)
test_info = get_dataset(args.test_path, norm=args.norm, mean_img=mean_img)
train = train_info.__getitem__(slice(0, train_info.__len__(), 1))
test = test_info.__getitem__(slice(0, test_info.__len__(), 1))

with open(args.param_path) as fp:
    param = json.load(fp)
with open(args.arch_path) as fp:
    config = json.load(fp)

predictor = discriminator(config, dr=param['dr'], bn=param['bn'])
model = custom_classifier(predictor=predictor)

if args.mname == 'None':
    args.mname = None

trainer = train_loop()
best_score = trainer(model, train, test, args.out_dir,
                     optname=args.optname, lr=param['lr'], rate=param['rate'],
                     lr_attr=args.lr_attr, gpu=args.gpu, bsize=param['bsize'],
                     test_bsize=args.test_bsize, esize=args.epoch,
                     mname=args.mname, weighting=param['weighting'],
                     l2=param['l2'])

with open(os.path.join(args.out_dir, 'best_score.json'), 'w') as fp:
    json.dump(best_score, fp)

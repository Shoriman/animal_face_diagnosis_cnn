import os
from glob import glob
import collections as cl
import numpy as np
from skimage import io

import chainer


class get_dataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_path, norm=0, mean_img=None, dtype=np.float32):
        self.class_dirs = [x for x in glob(data_path+'/*') if os.path.isdir(x)]
        self.class_dirs.sort()
        self.data_pairs = []
        self.file_order = []
        self.fpaths = []
        self.labels = []
        self.cls_names = cl.OrderedDict()
        for cls_index, cls_path in enumerate(self.class_dirs):
            self.cls_names[cls_index] = os.path.basename(cls_path)
            paths = glob(os.path.join(cls_path, '*.jpeg')) + \
                    glob(os.path.join(cls_path, '*.png')) + \
                    glob(os.path.join(cls_path, '*.tiff'))
            paths.sort()
            for file_path in paths:
                self.data_pairs.append((file_path, cls_index))
                self.fpaths.append(file_path)
                self.labels.append(cls_index)
                file_name = os.path.basename(file_path)
                file_index, _ = os.path.splitext(file_name)
                self.file_order.append(file_index)
        self.norm = norm
        self.dtype = dtype
        self.norm_method = {
            0: self.norm_8bit,
            -1: self.nothing,
        }
        self.mean_img = mean_img

    def __len__(self):
        return len(self.data_pairs)

    def get_example(self, i):
        path, label = self.data_pairs[i]
        img = io.imread(path).astype(self.dtype)
        img = self.norm_method[self.norm](img)
        if self.mean_img is not None:
            img -= self.mean_img
        img = img.transpose(2, 0, 1)
        return img, np.int32(label)

    def norm_8bit(self, img):
        return img/255

    def nothing(self, img):
        return img

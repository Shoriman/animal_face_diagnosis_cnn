import collections as cl

from chainer import cuda

from process_dataset.custom_iterator import custom_iterator
from evaluation.eval_func import aca, mca


class evaluator():
    def __init__(self, data, target, bsize=10):
        self._iterator = custom_iterator(data, batch_size=bsize,
                                         repeat=False, shuffle=False)
        self._target = target
        self.methods = {
            'aca': aca,
            'mca': mca,
        }

    def get_iterator(self):
        return self._iterator

    def get_target(self):
        return self._target

    def evaluate(self, metrics=['aca', 'mca'], device=None):
        results = cl.OrderedDict()
        if device >= 0:
            cuda.get_device(device).use()
            self._target.to_gpu()
        self._target.train = False
        for m in metrics:
            results[m] = float(self.methods[m](self, device))
        self._target.train = True
        return results

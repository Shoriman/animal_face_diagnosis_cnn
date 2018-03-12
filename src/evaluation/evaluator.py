import collections as cl

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
        self._target.train = False
        for m in metrics:
            results[m] = self.methods[m](device)
        self._target.train = True
        return results

import copy

from chainer import reporter as reporter_module
from chainer import variable
from chainer import cuda
from chainer.dataset import convert
from chainer.functions.evaluation import accuracy
import six
import numpy as np


def aca(evaluator, device=None):
    iterator = evaluator.get_iterator()
    target = evaluator.get_target()

    if hasattr(iterator, 'reset'):
        iterator.reset()
        it = iterator
    else:
        it = copy.copy(iterator)

    acc_lst = []
    for batch in it:
        in_arrays = convert.concat_examples(batch, device)
        if isinstance(in_arrays, tuple):
            in_vars = tuple(variable.Variable(x, volatile='on')
                            for x in in_arrays)
            y = target(in_vars[0]).data
            t = in_vars[1].data
            acc_lst.append(cuda.to_cpu(accuracy.accuracy(y, t).data))

    return np.mean(acc_lst)


def mca(evaluator, device=None):
    iterator = evaluator.get_iterator()
    target = evaluator.get_target()

    if hasattr(iterator, 'reset'):
        iterator.reset()
        it = iterator
    else:
        it = copy.copy(iterator)

    dic_summary = reporter_module.DictSummary()
    labels = it.get_labels()

    for batch in it:
        observation = {}
        in_arrays = convert.concat_examples(batch, device)
        if isinstance(in_arrays, tuple):
            in_vars = tuple(variable.Variable(x, volatile='on')
                            for x in in_arrays)

            xp = cuda.get_array_module(*in_vars)
            y = target(in_vars[0]).data
            t = in_vars[1].data
            pred = y.argmax(axis=1).reshape(t.shape)

            for l in labels:
                ind = xp.where(t == l)[0]
                if len(ind) == 0:
                    t_cnt = 0
                else:
                    t_cnt = len(xp.where(pred[ind] == l)[0])
                observation.update({l: t_cnt})

        dic_summary.add(observation)

    label_cnt = it.get_label_cnt()
    mca_score = xp.array([float(summary._x) / float(label_cnt[l]) for l, summary
                          in six.iteritems(dic_summary._summaries)]).mean()
    return cuda.to_cpu(mca_score)

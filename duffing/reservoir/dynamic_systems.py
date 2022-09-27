import itertools
from copy import deepcopy
from typing import Iterable
import numpy as np
from collections.abc import Iterable


def prepare_ns(ns):
    ns = list(ns)
    if not ns:
        return []
    elif isinstance(ns[0], Iterable):
        return expand_list_of_lists([[(inp, i) for i in ns_] for inp, ns_ in enumerate(ns)])
    else:
        return [(0, i) for i in ns]


def expand_list_of_lists(lofl):
    return list(itertools.chain(*lofl))


def get_order(prepared_nus, prepared_nys):
    return max([nys for _out, nys in prepared_nys] + [nus for _inp, nus in prepared_nus])


def get_size(prepared_ns):
    if prepared_ns:
        return max([out for out, _nys in prepared_ns]) + 1
    else:
        return 0


def formulate_problem(u, y, prepared_nus, prepared_nys):
    seq_len, n_seq, out_dim = y.shape
    order = get_order(prepared_nus, prepared_nys)
    regressors = []
    for inp, ny in prepared_nys:
        regressors.append(y[order - ny:seq_len - ny, :, inp])
    for inp, nu in prepared_nus:
        regressors.append(u[order - nu:seq_len - nu, :, inp])
    z = np.stack(regressors, axis=-1)
    return z


def simulate_nlss(fn, u, x0, k0=0):
    x = deepcopy(x0)
    # Compute first values
    y0, x_next = fn(x, u[0, ...], k0)  # uses reservoir state update equation
    # Instance output
    seq_len = u.shape[0]
    n_seq, out_dim = y0.shape
    y = np.zeros((seq_len, n_seq, out_dim))
    y[0, ...] = y0
    # iterate
    for k in range(1, seq_len):
        x = x_next
        y[k, ...], x_next = fn(x, u[k, ...], k0+k)
    return y, x_next


class FeedbackOutputs(object):
    def __init__(self, fn, prepared_nus, prepared_nys, n_states):
        self.order = get_order(prepared_nus, prepared_nys)
        self.n_outputs = get_size(prepared_nus)
        self.n_inputs = get_size(prepared_nus)
        self.fn = fn
        self.prepared_nys = prepared_nys
        self.prepared_nus = prepared_nus
        self.n_states = n_states

    def get_initial_state(self, u, y, x0):
        n_seq = u.shape[1]
        ys = y[:self.order, ...][::-1]
        # the second part will of the concatenation is so just the tensor has the right dimension, it
        # will be discarded in the first call...
        us = np.concatenate((u[:self.order, ...][::-1], np.zeros((1, n_seq, self.n_outputs))), axis=0)  # idk why
        return us, ys, x0

    def __call__(self, state, inp, k):
        us, ys, x = state
        # update us
        us[1:, ...] = us[:-1, ...]  # replace the last element with the previous one
        us[0, ...] = inp[:]  # replace the first element with the present input
        # the two lines above are used only to shift the input array (add the new input and preserve the last one)
        # Get z
        regressors = []
        for i, ny in self.prepared_nys:
            regressors.append(ys[ny-1, :, i])  # did not use the second replacement above?
        for i, nu in self.prepared_nus:
            regressors.append(us[nu, :, i])
        z = np.stack(regressors, axis=-1)  # pair of last output/input
        # Compute next state and output
        out, x_next = self.fn(x, z, k)
        # define ys
        ys[1:, ...] = ys[:-1, ...]
        ys[0, ...] = out[None, ...]
        return out, (us, ys, x_next)

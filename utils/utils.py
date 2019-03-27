#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

import numpy as np


def get_minibatches(data, size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list
                                        or type(data[0]) is np.ndarray)
    if list_data:
        data_size = len(data[0])
    else:
        data_size = len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)

    for start in np.arange(0, data_size, size):
        minibatch_indices = indices[start:start + size]
        yield [_minibatch(d, minibatch_indices) for d in data
               ] if list_data else _minibatch(data, minibatch_indices)


def _minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [
        data[i] for i in minibatch_idx
    ]

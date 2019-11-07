# -*- coding: utf-8 -*-
'''
Creat on 14/10/2019

Authors: shengzhixu

Email: 

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.signal import convolve, deconvolve
from numpy.fft import fft, fftshift, ifft
from PyMat.utils import next_pow


x = np.array([1, 2, 3, 4, 2, 1, 10])
h = np.array([1, 1, 1, 1, 1, 1, 2, 2, 3, 4])
# x = np.random.randn(100)
# h = np.random.randn(100)

def _conv_fft(x, h, mode='same'):
    N = next_pow(np.max([x.size, h.size]))
    X = fft(x, N)
    H = fft(h, N)
    Y = X*H
    y = ifft(Y)


    if mode == 'same':
        if h.size % 2 == 1:
            starting_point = h.size // 2
        else:
            starting_point = h.size // 2 - 1
        return y[starting_point:starting_point+x.size]
    else:
        return y


def _deconv_fft(y, h):
    N = next_pow(np.max([y.size, h.size]))
    Y = fft(y, N)
    H = fft(h, N)
    X = Y/H
    x = ifft(X)

    _size = np.sum(abs(x) >= 1e-8)
    return x[0:_size]


y = convolve(x, h, 'same')
print('y', y)

yy = _conv_fft(x, h, mode=None)
print('yy', yy.real)

xx = _deconv_fft(yy, h)
print(xx.real)


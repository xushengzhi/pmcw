# -*- coding: utf-8 -*-
'''
Creat on 14/10/2019

Authors: shengzhixu

Email: 

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import exp, pi, log2
from numpy.fft import fft, fftshift, ifftshift, ifft
from scipy import signal
from scipy import convolve

from PyMat.SysParas import *

# EFFECTIVE_LENGTH = 400_000


# %% Load match code
def load_matched_code(code_file, verbose=False, win_func=None):
    # f = np.linspace(-0.5, 0.5, EFFECTIVE_LENGTH, endpoint=False) * SAMPLING_FREQUENCY
    demix_wave = exp(-2j * pi * INTERMEDIATE_FREQUENCY * np.arange(EFFECTIVE_LENGTH) / SAMPLING_FREQUENCY)
    zeroing_length = int(EFFECTIVE_LENGTH / 16)
    wave = np.loadtxt(code_file, delimiter=',')[0::3, 0]
    wave_fft = fftshift(fft(wave[0:EFFECTIVE_LENGTH] * demix_wave))
    if win_func is None:
        pass
    elif win_func is 'rect':
        wave_fft[0:7 * zeroing_length] = 0
        wave_fft[9 * zeroing_length::] = 0
    else:
        win_func = eval('signal.windows.' + win_func)
        print("The windowing function {} is applied for slow time".format(win_func.__name__))
        win = win_func(wave_fft.size)
        wave_fft = wave_fft * win

    w = ifft(ifftshift(wave_fft))

    # TODO: resample the data from 1.2GHz => SAMPLING_FREQUENCY

    if verbose:
        plt.figure()
        plt.plot(w.real, label='real part')
        plt.plot(w.imag, label='imaginary part')
        plt.legend(loc=1)

    return w


def next_pow(n):
    n = 2**np.ceil(log2(n)+1)
    return int(n)


def svdinv(R):
    U, A, V = np.linalg.svd(R, full_matrices=False)
    return U @ np.diag(1/A) @ V


def conv_fft(x, h, mode='same'):
    M = np.max([x.size, h.size])
    K = np.min([x.size, h.size])
    N = next_pow(M)
    X = fft(x, N)
    H = fft(h, N)
    Y = X*H
    y = ifft(Y)

    if K % 2 == 1:
        starting_point = (K) // 2
    else:
        starting_point = (K-1) // 2
    if mode == 'same':
        return y[starting_point:starting_point+M]
    else:
        return y


def deconv_fft(y, h):
    N = next_pow(np.max([y.size, h.size]))
    Y = fft(y, N)
    H = fft(h, N)
    X = Y/H
    x = ifft(X)

    # _size = np.sum(abs(x) >= 1e-8)
    return x

if __name__ == '__main__':
    a = np.random.randint(0, 10, size=np.random.randint(2, 15))
    b = np.random.randint(0, 10, size=np.random.randint(2, 15))
    # a = np.array([6, 6, 4, 6, 9, 8, 7])
    # b = np.array([4, 9, 0, 9, 7, 8])

    print("convolve(a, b): {}".format(convolve(a, b, 'same')))
    print("conv_fft(a, b): {}".format(conv_fft(a, b, 'same').real))

    c = conv_fft(a, b, 'full')
    d = deconv_fft(c, b)
    print('a: {}'.format(a))
    print('d: {}'.format(d.real[0:a.size]))
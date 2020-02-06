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

from SysParas import *

# EFFECTIVE_LENGTH = 400_000


# %% Load match code
def load_matched_code(code_file, verbose=False, win_func=None, compensated=True):
    # f = np.linspace(-0.5, 0.5, EFFECTIVE_LENGTH, endpoint=False) * SAMPLING_FREQUENCY
    demix_wave = exp(-2j * pi * INTERMEDIATE_FREQUENCY * np.arange(EFFECTIVE_LENGTH) / SAMPLING_FREQUENCY)
    zeroing_length = int(EFFECTIVE_LENGTH / 16)
    wave = np.loadtxt(code_file, delimiter=',')[:, 0]
    if compensated:
        w = wave[0:EFFECTIVE_LENGTH] * demix_wave
    else:
        w = wave[0:EFFECTIVE_LENGTH]

    if win_func is None:
        pass
    else:
        wave_fft = fftshift(fft(w))
        if win_func is 'rect':
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


def conv_circ( signal, ker, mode=None ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''

    return np.real(np.fft.ifftshift(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) )))


def dB(d, normalize=False, mode='s'):

    db_value = 10*np.log10(abs(d) + 1e-40)

    if mode == 'm':
        db_value = 2*db_value

    if normalize:
        return db_value - np.max(db_value)
    else:
        return db_value


def roll_zeropad(a, shift, axis=None, mode=None, *args ,**kwargs):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll_zeropad(x, 2)
    array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll_zeropad(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> roll_zeropad(x2, 1)
    array([[0, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2)
    array([[2, 3, 4, 5, 6],
           [7, 8, 9, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> roll_zeropad(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=1)
    array([[0, 0, 1, 2, 3],
           [0, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2, axis=1)
    array([[2, 3, 4, 0, 0],
           [7, 8, 9, 0, 0]])

    >>> roll_zeropad(x2, 50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, -50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 0)
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

    """
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def random_code(size):
    '''
    Generate random codes according to the Gaussian distribution
    :param size:
    :return:
    '''
    x = np.random.randn(size)
    x[x>0] = 1
    x[x<=0] = -1

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
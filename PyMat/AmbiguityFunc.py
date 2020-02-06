# -*- coding: utf-8 -*-
'''
Creat on 28/01/2020

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

import numpy as np
from numpy import exp, sqrt, log10, pi
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix
from scipy.constants import speed_of_light as c
import sys
from utils import roll_zeropad, conv_circ
from tqdm import tqdm


#%%
def ambiguity_function(data, f_vec, fs,
                       normalize=True, mode='zero-padding', *args, **kwargs):
    '''
    Narrowband Ambiguity Function
    :param data: 1D numpy array, vectorized time sampled data
    :param tau_vec: time delay vector
    :param f_vec: Doppler frequency vector
    :param mode: time delay mode, 'shift'(default) or 'zero-padding'
    :return: ambiguity function with shape of len(tau_vec) X len(f_vec)
    '''

    if normalize:
        data = data/np.sqrt(abs(np.sum(data * data.conj())))    # normalize
    M = len(data)                                           # time index
    K = len(f_vec)                                          # frequency index
    t = np.arange(M) / fs                                   # time vector

    E = exp(-2j*pi*f_vec.reshape(K, 1) @ t.reshape(1, M))   # K X M
    if mode == 'zero-padding':
        P = M-1
        tau_ind = np.arange(-M+1, M)
        U = np.zeros((M, M + P), dtype=complex)  # M X (M+P)
        U[0:M, 0:M] = np.diag(data)
        U = dia_matrix(U)
        T = np.zeros((M + P, 2*M+1), dtype=complex)                 # (M+P) X N
        for i, tau in enumerate(tau_ind):                       # delay shift data (conjugate)
            if tau >= 0:
                T[tau:tau+M, i] = data.conj()
            else:
                T[0:(M+tau), i] = data[-tau::].conj()

    elif mode == 'shift':
        tau_ind = np.arange(-M+1, M)
        U = dia_matrix(np.diag(data))                       # M X M
        T = np.zeros((M, 2*M+1), dtype=complex)                 # M X N
        for i, tau in enumerate(tau_ind):
            T[:, i] = np.roll(data.conj(), tau)

    else:
        print('MODE value can be only "zero-padding" or "shift"')
        raise ValueError

    print('Size of T matrix is {}MB.'.format(sys.getsizeof(T)/1024/1024))
    af = E @ U @ T
    
    return af


def ambiguity_function2(data, f_vec, fs,
                       normalize=True, mode='zero-padding', *args, **kwargs):
    if normalize:
        data = data/np.sqrt(abs(np.sum(data * data.conj())))    # normalize
    M = len(data)                                           # time index
    K = len(f_vec)                                          # frequency index
    t = np.arange(M) / fs                                   # time vector

    af = np.zeros((K, 2 * M - 1), dtype=complex)

    for i, fd in tqdm(enumerate(f_vec)):
        E = exp(-2j*pi*fd*t)        # M
        U = data * E                # M

        if mode == 'zero-padding':
            af[i, :] = np.convolve(U, data[::-1].conj(), mode='full')
        elif mode == 'shift':
            Ure = np.roll(np.concatenate((U, U)), (M)//2)
            af[i, :] = np.convolve(Ure, data[::-1].conj(), mode='same')[1::]

    return af

# %% main
if __name__ == '__main__':

    # %% LFM
    # f0 = 0e9
    # B = 100e3
    # T = 0.0001
    # mu = B/T
    # fs = 200e3
    # N = int(T*fs)
    # t = np.arange(N)/fs
    # trans = exp(2j*pi*(f0*t + 0.5*mu*t**2))  + np.random.randn(N) * 0.1
    # tau_vec = np.arange(-N+1, N)/fs
    # f_vec = np.linspace(-0.5, 0.5, 128, endpoint=False)*fs
    # # af = ambiguity_function2(trans, f_vec, fs, mode='shift')
    # af = ambiguity_function(trans, f_vec, fs, mode='zero-padding')

    # %%random codes
    # fs = 1
    # code = np.random.rand(128)
    # trans = np.zeros_like(code)
    # trans[code>0.5] = -1
    # trans[code<=0.5] = 1
    # # code += np.random.randn(128) * 0.01
    # f_vec = np.linspace(-0.5, 0.5, 256, endpoint=False)
    # af = ambiguity_function(trans, f_vec, fs, mode='shift')

    # %%pulse
    # code = np.zeros((64, ))
    # fs = 1
    # code[30:40] = 1
    # code += np.random.randn(64) * 0.001
    # tau_vec=np.arange(-32, 33)
    # f_vec = np.linspace(-0.5, 0.5, 128)
    # af = ambiguity_function(code, tau_vec, f_vec, fs)

    # %%BarkerCode
    code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    code = np.kron(code, np.ones((1, )))
    fs = 1
    f_vec = np.linspace(-0.5, 0.5, 256)
    af = ambiguity_function(code, f_vec, fs, mode='zero-padding')
    # af = ambiguity_function(code, f_vec, fs, mode='shift')

    # %% FrankCode
    # phase = [0, 0, 0, 0, 0, pi/2, pi, 1.5*pi, 0, pi, 0, pi, 0, 1.5*pi, pi, pi/2]
    # code = exp(1j*np.array(phase))
    # fs = 1
    # f_vec = np.linspace(-0.5, 0.5, 256)
    # tau_vec = np.arange(-8, 9)
    # af = ambiguity_function(code, tau_vec, f_vec, fs)

    # %%
    plt.figure()

    # plt.imshow((abs(af)), aspect='auto', cmap='hot_r')
    plt.contour((abs(af)), cmap='jet', levels=15)
    plt.ylabel(r'$f/f_s$')
    plt.xlabel(r'$\tau \times f_s$')
    cb = plt.colorbar()

    # plt.figure();
    # plt.plot(abs(af[:, :]))
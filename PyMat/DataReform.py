# -*- coding: utf-8 -*-
'''
Creat on 28/09/2019

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.io import savemat
from scipy import signal
from numpy.fft import fftshift, fft, ifft, ifftshift, fft2
from numpy import log10, exp, pi
from numpy.linalg import pinv

from PyMat.SysParas import INTERMEDIATE_FREQUENCY, SAMPLING_FREQUENCY, PERIOD_DURATION
from PyMat.utils import load_matched_code, next_pow, svdinv


def data_reform(file,
                save_data=False,
                n_block_to_process=65,
                filter=True,
                verbose=False,
                win_func=None,
                hfunc=True,
                filter_with_error = False):

    '''
    Reform the original data
    :param file: file path with .bin file
    :return: data and tdata
    '''

    # % Load file
    with open(file, 'rb') as fid:
        info = np.fromfile(fid, count=2, dtype=np.int32)
        print(info[0])
        block_to_skip = 1

        if block_to_skip > 0:
            for i in range(block_to_skip):
                A = np.fromfile(fid, count=info[0], dtype=np.int16)
                print(A.shape)

        A = np.fromfile(fid, count=info[0], dtype=np.int16)
        A = np.delete(A, np.s_[0:4])
        print(A.shape)
        print("Starting to read data ...")
        for _ in tqdm(range(n_block_to_process-1)):
            B = np.fromfile(fid, count=info[0], dtype=np.int16)
            np.delete(B, np.s_[0:4])
            A = np.append(A, B)

    # % reformation
    print("size of A : {}".format(A.size))
    B = A.reshape([A.size//2, 2])

    trans = B[:, 0]
    recei = B[:, 1]

    # plt.figure()
    # plt.plot(recei.real)

    starting_point = 200962     # 202796
    slowtime_length = (400000 - 4)//int(1e-3/PERIOD_DURATION)
    slowtime = n_block_to_process // (2//int(1e-3/PERIOD_DURATION)) - 1*(PERIOD_DURATION<1e-3)
    end_point = starting_point + slowtime_length * slowtime

    trans = trans[starting_point:end_point]
    recei = recei[starting_point:end_point]

    if verbose:
        plt.plot(trans)
        plt.title("Transmitted Signals (only real)")

    # demix_wave = exp(-2j * pi * INTERMEDIATE_FREQUENCY * np.arange(recei.size) / SAMPLING_FREQUENCY)
    # recei = recei * demix_wave
    # trans = trans * demix_wave

    if hfunc:
        print('Generated wave loading ...')
        wave = np.loadtxt('pmcw_waveform.txt', delimiter=',')[0::3, 0]
        W = np.tile(wave, [slowtime, 1])    # slow * fast
        print('Generated wave loaded!')

    recei = recei.reshape([slowtime, slowtime_length]).T.astype('complex')
    trans = trans.reshape([slowtime, slowtime_length]).T.astype('complex')

    if filter:
        print("Filter starts ...")
        if filter_with_error:
            trans, H = filtering(trans, compensated=True, verbose=verbose, win_func=win_func, mode='trans', t_mat=W)
            recei, _ = filtering(recei, compensated=True, verbose=verbose, win_func=win_func, mode='recei', t_mat=H)
        else:
            trans, H = filtering(trans, compensated=True, verbose=verbose, win_func=win_func, mode=None, t_mat=W)
            recei, _ = filtering(recei, compensated=True, verbose=verbose, win_func=win_func, mode=None, t_mat=H)
        print("Filter finished!")

    data = recei.reshape([slowtime, slowtime_length]).T.astype('complex')
    tdata = trans.reshape([slowtime, slowtime_length]).T.astype('complex')

    # # filter for each slow time
    # if filter:
    #     print("Filter starts ...")
    #     for i in range(slowtime):
    #         data[:, i] = _filtering(data[:, i], compensated=True)
    #         tdata[:, i] = _filtering(tdata[:, i], compensated=True)
    #     print("Filter finished!")

    if verbose:
        plt.figure()
        plt.imshow(tdata.real, aspect='auto', cmap=CMAP)
        plt.colorbar()
    print("Data are read!")

    # % Save file for further process
    if save_data:
        mat_file_name = file.split('/')[-1][:-3]+'mat'
        savemat(mat_file_name, mdict={'data': data, 'tdata': tdata})
        print('Data are saved as: '.format(mat_file_name))

    return data, tdata


def filtering(data,
               compensated=True,
               verbose=False,
               win_func=None,
               mode=None,
               t_mat=None,
               fs=SAMPLING_FREQUENCY,
               fi=INTERMEDIATE_FREQUENCY):
    # system transmit error problem
    H = None

    S, F = data.shape
    if mode=='trans':
        print('Trans mode started ...')
        H = (fft2(t_mat, [next_pow(S), next_pow(F)])) @ svdinv(fft2(data, [next_pow(S), next_pow(F)])).T.conj()
        print('Trans mode finished')
    elif mode=='recei':
        print('Recei mode started ...')
        data = ifft((fft2(data, [next_pow(S), next_pow(F)])) @ t_mat)[0:S, 0:F].flatten()
        print('Recei mode finished')
    else:
        pass

    # N = data.size
    # if mode=='trans':
    #     print('Trans mode started ...')
    #     H = fft(data, next_pow(N)) / fft(t_mat, next_pow(N))
    #     print('Trans mode finished')
    # elif mode=='recei':
    #     print('Recei mode started ...')
    #     data = ifft((fft(data, next_pow(N)) / t_mat))[0:N]
    #     print('Recei mode finished')
    # else:
    #     pass

    # compensation
    if compensated:
        demix_wave = exp(-2j * pi * fi * np.arange(data.size) / fs)
    else:
        demix_wave = np.ones_like(data)
    data_shift_fft = fftshift(fft(data * demix_wave))
    zeroing_length = data.size // 16
    data_filtered = data_shift_fft.copy()

    # window function
    if win_func is None:
        pass
    elif win_func is 'rect':
        print("The Rectangular windowing function is applied for slow time")
        data_filtered[0:6 * zeroing_length] = 0
        data_filtered[10 * zeroing_length::] = 0
    else:
        win_func = eval('signal.windows.' + win_func)
        print("The windowing function {} is applied for slow time".format(win_func.__name__))
        win = win_func(data.size)
        data_filtered = data_filtered * win

    if verbose:
        data_fft = fftshift(fft(data))
        data_fft_db_max = np.max(20 * log10(abs(data_fft)))
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
        f = np.linspace(-0.5, 0.5, data_fft.size, endpoint=False) * fs

        axs[0].plot(f, 20 * log10(abs(data_fft) + 1e-20))
        axs[0].set_title("Original spectrum")
        axs[0].grid(ls=':')
        axs[0].set_ylim([data_fft_db_max-150, data_fft_db_max+10])
        axs[1].plot(f, 20 * log10(abs(data_shift_fft) + 1e-20))
        axs[1].set_title("Shifted spectrum")
        axs[1].grid(ls=':')
        axs[2].plot(f, 20 * log10(abs(data_filtered) + 1e-20))
        axs[2].set_title("Filtered spectrum")
        axs[2].grid(ls=':')
        print('f with max amplitude: {}'.format(f[np.max(abs(data_filtered)) == abs(data_filtered)]))

    dataf = ifft(ifftshift(data_filtered))

    return dataf, H


# %% Main func
if __name__ == '__main__':
    # path = '/Volumes/Personal/Backup/PMCWPARSAXData/cars/'
    # file = 'HH_20190919134453.bin'
    CMAP = 'jet'

    path = '/Volumes/Personal/Backup/PMCWPARSAXData/chimney/'
    file ='HH_20190919075847.bin'
    target = 'chimney'

    path =  '/Volumes/Personal/Backup/PMCWPARSAXData/A13/'
    channel = 'VV'
    file = channel + '_20191009081112.bin'
    PERIOD_DURATION = 0.5e-3

    INTERMEDIATE_FREQUENCY = 125_000_000
    SAMPLING_FREQUENCY = 399_996_327 + 230 + 50 + 41 + 20 + 21 + 30 + 12
    SAMPLING_FREQUENCY = 400e6

    data, tdata = data_reform(path + file,
                              verbose=True,
                              filter=True,
                              n_block_to_process=3,
                              win_func='rect',
                              filter_with_error=True)

    plt.figure()
    # plt.plot(data.T.flatten().imag, label='imaginary part')
    plt.plot(tdata.T.flatten().real[::1], label='real part')
    plt.legend(loc=1)


    f = np.linspace(-1/2/PERIOD_DURATION, 1/2/PERIOD_DURATION, 128, endpoint=False)
    plt.figure()
    plt.imshow(abs((fft(tdata[0:4000:8, :], n=128, axis=-1))),
               aspect='auto',
               cmap=CMAP,
               extent=[f[0], f[-1], 0, 200]
               )

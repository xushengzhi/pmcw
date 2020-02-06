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

from SysParas import INTERMEDIATE_FREQUENCY, SAMPLING_FREQUENCY, PERIOD_DURATION, CMAP
from utils import load_matched_code, next_pow, svdinv


def data_reform(file,
                save_data=False,
                n_block_to_process=65,
                filter=True,
                verbose=False,
                win_func=None):

    '''
    Reform the original data
    :param file: file path with .bin file
    :return: data and tdata (received data, transmitted data)
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

    starting_point = 203064           #200962     # 202796
    slowtime_length = (400000 - 2)//int(1e-3/PERIOD_DURATION)
    slowtime = int(n_block_to_process // (2/int(1e-3/PERIOD_DURATION)) - 1*(PERIOD_DURATION<1e-3))
    end_point = starting_point + slowtime_length * slowtime

    trans = trans[starting_point:end_point]
    recei = recei[starting_point:end_point]
    trans[trans>5000] = 0

    if verbose:
        plt.figure()
        plt.plot(trans)
        plt.title("Transmitted Signals (only real)")

    recei = recei.reshape([slowtime, slowtime_length]).T.astype('complex')
    trans = trans.reshape([slowtime, slowtime_length]).T.astype('complex')

    if verbose:
        plt.figure()
        plt.imshow(trans.real, aspect='auto', cmap=CMAP)
        plt.colorbar()
        plt.title('Before Alignment')
        plt.xlabel('Slow-time index')
        plt.ylabel('Samples')
    trans[:, ::2]  = np.roll(trans[:, ::2], -1, axis=0)
    recei[:, ::2]  = np.roll(recei[:, ::2], -1, axis=0)

    for i in range(slowtime//2):
        if i==2:
            length_shift = 1750  #1923 * (i-1)
            trans[0:length_shift, i * 2] = np.roll(trans[0:length_shift, i * 2], 2, axis=0)
            recei[0:length_shift, i * 2] = np.roll(recei[0:length_shift, i * 2], 2, axis=0)
        elif i>2:
            length_shift = 1750 + 1923 * (i-2)
            trans[0:length_shift, i * 2] = np.roll(trans[0:length_shift, i * 2], 2, axis=0)
            recei[0:length_shift, i * 2] = np.roll(recei[0:length_shift, i * 2], 2, axis=0)
        else:
            pass

    # Ntrans = trans.copy()
    # Ntrans[:, ::2] = np.roll(trans[:, ::2], -2, axis=0)
    # plt.figure()
    # plt.imshow((abs(Ntrans.real)), aspect='auto')
    # if verbose:
    #     plt.figure()
    #     plt.imshow((abs(Ntrans.real)), aspect='auto')

    if filter:
        print("Filter starts ...")

        trans = filtering(trans, compensated=True, verbose=verbose, win_func=win_func)
        recei = filtering(recei, compensated=True, verbose=verbose, win_func=win_func)
        print("Filter finished!")

        data = recei.reshape([slowtime, slowtime_length]).T.astype('complex')
        tdata = trans.reshape([slowtime, slowtime_length]).T.astype('complex')
    else:
        data = recei
        tdata = trans

    if verbose:
        plt.figure()
        plt.imshow(tdata.real, aspect='auto', cmap=CMAP)
        plt.colorbar()
        plt.title('After Alignment')
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
               fs=SAMPLING_FREQUENCY,
               fi=INTERMEDIATE_FREQUENCY):
    # system transmit error problem

    # compensation
    if compensated:
        demix_wave = exp(-2j * pi * fi * np.arange(data.shape[0]) / fs)
        demix_wave = np.tile(demix_wave, data.shape[1])
    else:
        demix_wave = np.ones_like(data.flatten())
    data_shift_fft = fftshift(fft(data.T.flatten() * demix_wave))
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
        data_fft = fftshift(fft(data.T.flatten()))
        data_fft_db_max = np.max(20 * log10(abs(data_fft)))
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
        f = np.linspace(-0.5, 0.5, data_fft.size, endpoint=False) * fs/1e6

        axs[0].plot(f, 20 * log10(abs(data_fft) + 1e-20))
        # axs[0].set_title("Original spectrum")
        axs[0].grid(ls=':')
        axs[0].set_ylabel('dB')
        axs[0].set_ylim([data_fft_db_max-150, data_fft_db_max+10])
        axs[1].plot(f, 20 * log10(abs(data_shift_fft) + 1e-20))
        # axs[1].set_title("Shifted spectrum")
        axs[1].grid(ls=':')
        axs[1].set_ylabel('dB')
        axs[2].plot(f, 20 * log10(abs(data_filtered) + 1e-20))
        # axs[2].set_title("Filtered spectrum")
        axs[2].grid(ls=':')
        axs[2].set_ylabel('dB')
        axs[2].set_xlabel('Frequency (MHz)')
        print('f with max amplitude: {}'.format(f[np.max(abs(data_filtered)) == abs(data_filtered)]))

    dataf = ifft(ifftshift(data_filtered))

    return dataf


# %% Main func
if __name__ == '__main__':
    # path = '/Volumes/Personal/Backup/PMCWPARSAXData/cars/'
    # file = 'HH_20190919134453.bin'
    path = '/Volumes/Personal/ExternalDrive/Backup/PMCWPARSAXData/sync_clock_data_a13/'
    file = 'VV/VV_20191126093852.bin'
    # file = 'VV/VV_20191126093725.bin'
    PERIOD_DURATION = 1e-3/2
    save_fig = False

    # path = '/Volumes/Personal/Backup/PMCWPARSAXData/chimney/'
    # file ='HH_20190919075847.bin'
    # target = 'chimney'

    # path =  '/Volumes/Personal/Backup/PMCWPARSAXData/A13/'
    # channel = 'VV'
    # file = channel + '_20191009081112.bin'
    # PERIOD_DURATION = 0.5e-3

    INTERMEDIATE_FREQUENCY = 125_000_000
    SAMPLING_FREQUENCY = 399_996_327 # + 230 + 50 + 41 + 20 + 21 + 30 + 12
    # SAMPLING_FREQUENCY = 400e6

    data, tdata = data_reform(path + file,
                              verbose=True,
                              filter=True,
                              n_block_to_process=11,
                              win_func='rect')

    # plt.figure(figsize=[12,3])
    # plt.plot(tdata.T.flatten().real, label='imaginary part')
    # # plt.plot(tdata[:, -1].real, label='real part')
    # plt.xlabel('sampling index')
    # plt.ylabel('amplitude')
    # plt.title('Trans Sig (recorded)')
    # # plt.ylim([-450, 450])
    # plt.legend(loc=1)
    # if save_fig:
    #     plt.savefig('trans_sig_recorded_{}ms.png'.format(str(PERIOD_DURATION*1e3)), dpi=300)

    # %%
    plt.figure(figsize=[15, 3])
    # plt.vlines(131072, -100, 100, colors='r', linewidth=2)
    plt.plot(data[:, 0:10].T.flatten().real, label='real part')
    # plt.plot(data[:, -1].real, label='real part')
    plt.xlabel('sampling index')
    plt.ylabel('amplitude')
    # plt.title('Receiv Sig')
    # plt.ylim([-450, 450])
    plt.legend(loc=1)
    plt.tight_layout()
    plt.vlines(131072, 0, 130, colors='r', linewidth=2)
    plt.vlines(200095, 0, 130, colors='r', linewidth=2)
    plt.hlines(125, 131072, 200095, colors='r', linestyles=':', lw=2)
    plt.text(142100, 100, 'nulls')
    if save_fig:
        plt.savefig('recei_sig_recorded_{}ms.png'.format(str(PERIOD_DURATION*1e3)), dpi=300)


    # f = np.linspace(-1/2/PERIOD_DURATION, 1/2/PERIOD_DURATION, 128, endpoint=False)
    # plt.figure()
    # plt.imshow(abs(fftshift(fft(tdata[10000:12000, :], n=128, axis=-1))),
    #            aspect='auto',
    #            cmap=CMAP,
    #            extent=[f[0], f[-1], 0, 200]
    #            )
    # if save_fig:
    #     plt.savefig('Doppler {}ms.png'.format(str(PERIOD_DURATION*1e3)), dpi=300)
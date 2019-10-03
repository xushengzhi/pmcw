# -*- coding: utf-8 -*-
'''
Creat on 28/09/2019

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.io import savemat
from numpy.fft import fftshift, fft, ifft, ifftshift
from numpy import log10, exp, pi

from PyMat.SysParas import CENTRE_FREQUENCY, SAMPLING_FREQUENCY

CMAP = 'jet'

def data_reform(file,
                save_data=False,
                n_block_to_process=65,
                filter=True,
                verbose=False):
    '''
    Reform the original data
    :param file: file path with .bin file
    :return: data and tdata
    '''
    # %% Load file
    with open(file, 'rb') as fid:
        info = np.fromfile(fid, count=2, dtype=np.int32)

        print(info[0])

        block_to_skip = 1
        # n_block_to_process = 65

        if block_to_skip >0:
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

    # %% reformation
    print("size of A : {}".format(A.size))
    B = A.reshape([A.size//2, 2])

    trans = B[:, 0]
    recei = B[:, 1]

    starting_point = 200962
    slowtime_length = 399998

    slowtime = n_block_to_process//2

    if filter:
        print("Filter starts ...")
        data_fft = fftshift(fft(recei))
        demix_wave = exp(-2j * pi * CENTRE_FREQUENCY * np.arange(recei.size) / SAMPLING_FREQUENCY)
        data_shift_fft = fftshift(fft(recei*demix_wave))
        zeroing_length = recei.size//8
        data_filtered = data_shift_fft.copy()
        data_filtered[0:3*zeroing_length] = 0
        data_filtered[5*zeroing_length::] = 0

        if verbose:
            fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
            axs[0].plot(20*log10(abs(data_fft) + 1e-20))
            axs[0].set_title("Original spectrum")
            axs[1].plot(20*log10(abs(data_shift_fft) + 1e-20))
            axs[1].set_title("Shifted spectrum")
            axs[2].plot(20*log10(abs(data_filtered) + 1e-20))
            axs[2].set_title("Filtered spectrum")

        recei = ifft(ifftshift(data_filtered))
        print("Filter finished!")

    end_point = starting_point + slowtime_length * slowtime
    data = recei[starting_point:end_point].reshape([slowtime, slowtime_length]).T
    tdata = trans[starting_point:end_point].reshape([slowtime, slowtime_length]).T

    if verbose:
        plt.figure()
        plt.imshow(tdata, aspect='auto', cmap=CMAP)
        plt.colorbar()
    print("Data are read!")
    # %% Save file for further process
    if save_data:
        mat_file_name = file.split('/')[-1][:-3]+'mat'
        savemat(mat_file_name, mdict={'data':data, 'tdata':tdata})
        print('Data are saved as: '.format(mat_file_name))

    return data, tdata


# %% Main func
if __name__ == '__main__':
    file = '/Users/shengzhixu/Documents/ExperimentalData/pmcw/cars/HH_20190919134453.bin'
    data_reform(file, verbose=True)
# -*- coding: utf-8 -*-
'''
Creat on 30/09/2019

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.io import loadmat
from tqdm import tqdm
from scipy.constants import speed_of_light as C
from numpy import pi, exp, log10
from numpy.fft import fft, fftshift, ifftshift, ifft
from scipy.signal import convolve

from PyMat.DataReform import data_reform
from PyMat.SysParas import *
from PyMat.utils import load_matched_code


# %% fast-time correlation
def fast_time_correlation(data, matched_code, downsampling_rate=1, conv_mode='same'):

    print("\nFast-time correlation started ...")

    data_downsampling = data[::downsampling_rate, :]
    if conv_mode is 'same':
        range_data = np.zeros_like(data_downsampling)
    else:
        range_data = np.zeros((data_downsampling.shape[0]*2-1, data_downsampling.shape[1]), dtype='complex')

    for i in tqdm(range(data.shape[1])):
        if matched_code.ndim == 1:
            matched_code_downsampling = matched_code[::-downsampling_rate].conj()
            range_data[:, i] = convolve(data_downsampling[:, i], matched_code_downsampling, mode=conv_mode)
        else:
            matched_code_downsampling = matched_code[::-downsampling_rate, ...].conj()
            range_data[:, i] = convolve(data_downsampling[:, i], matched_code_downsampling[:, i], mode=conv_mode)

    print("Fast-time correlation finished!")

    return range_data


# %% slow-time FFT
def slow_time_fft(data, win_func=None, fft_zoom=1, shifted=False):
    print("\nSlow-time FFT started ...")
    win = np.ones_like(data)
    if win_func is not None:
        try:
            win_func = eval('sp.signal.windows.' + win_func)
            print("The windowing function {} is applied for slow time".format(win_func.__name__))
            win = np.ones((data.shape[0], 1)).dot(win_func(data.shape[1]).reshape(1, data.shape[1]))
        except:
            print('Please provide a valid windowing function: e.g. win_func="hann"')
    doppler_data = fft(data * win, axis=-1, n=data.shape[1]*fft_zoom)
    if shifted:
        doppler_data = fftshift(doppler_data, axes=-1)
    print("Slow-time FFT finished!")
    return doppler_data


# %% images process
def range_doppler_show(data,
                       start_plot=0,
                       end_plot=8000,
                       downsampling_rate=1,
                       clim=40,
                       dB = True,
                       normalize=True,
                       cmap='jet',
                       title="Range_Doppler",
                       save_fig=False):
    print("\nRange-Doppler imaging started ...")
    dr = C / 2 / SAMPLING_FREQUENCY * downsampling_rate
    x = np.arange(EFFECTIVE_LENGTH) * dr
    vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION
    v = np.linspace(-vm, vm, data.shape[1], endpoint=False)
    plt.figure()
    if dB:
        data_db = 20 * log10(abs(np.flipud(data[start_plot:end_plot, :]))+1e-40)
    else:
        data_db = np.flipud(data[start_plot:end_plot, :])

    max_db = np.max(data_db)
    print("Maximum amplitude: {:.4}dB".format(max_db))
    if normalize:
        data_db = data_db - max_db
        max_db = 0
    plt.imshow(data_db,
               aspect='auto',
               cmap=cmap,
               extent=[v[0], v[-1], x[start_plot], x[end_plot]])
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Range (m)')
    plt.clim([max_db-clim, max_db])
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("(dB)")
    if save_fig:
        plt.savefig("{}.png".format(title), dpi=300)
    print("Range-Doppler imaging Finished")


# %% Doppler compensation
def doppler_compensation(data, fft_zoom=1, downsampling_rate=1):
    print("\nDoppler shifts compensation started ...")
    vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION
    doppler_cells = data.shape[1]*fft_zoom
    v_vector = np.linspace(-vm, vm, doppler_cells, endpoint=False).reshape(1, doppler_cells)
    print("Maximum Doppler shifts is {}Hz".format(1/2/PERIOD_DURATION))
    omega = -2j*pi*2/C*CARRIER_FREQUENCY/SAMPLING_FREQUENCY*downsampling_rate
    compensation_matrix = exp(omega * np.arange(data.shape[0]).reshape(data.shape[0], 1).dot(v_vector))
    print("Doppler shifts compensation finished!")
    #
    # plt.figure()
    # plt.imshow(np.angle(compensation_matrix), aspect='auto', cmap=CMAP)

    return data * compensation_matrix


if __name__ == '__main__':
    # %% Pre-Setting
    save_fig = False
    CMAP = 'jet'
    win_func = 'hann'
    matched = True

    if matched:
        match_code = 'pmcw8192.txt'
        match_code = 'pmcw_waveform.txt'
    else:
        match_code = 'pmcw_waveform_code2.txt'

    # %% Load data
    # data = loadmat('data.mat')
    # recei = data['data']
    # trans = data['tdata']
    path =  '/Volumes/Personal/Backup/PMCWPARSAXData/cars/'
    file ='HH_20190919134453.bin'
    target = 'cars'

    # path = '/Volumes/Personal/Backup/PMCWPARSAXData/chimney/'
    # file ='HH_20190919075858.bin'
    # target = 'chimney'

    path =  '/Volumes/Personal/Backup/PMCWPARSAXData/A13/'
    file = 'VV_20191009081112.bin'
    target = 'vehicles'
    PERIOD_DURATION = 0.5e-3

    N_Block = 141
    recei, trans = data_reform(path + file,
                               n_block_to_process=N_Block,
                               verbose=False,
                               filter=True,
                               win_func='rect')

    fast_time, slow_time = recei.shape
    global EFFECTIVE_LENGTH
    EFFECTIVE_LENGTH = 262144//int(1e-3/PERIOD_DURATION)
    recei_rec = recei[0:EFFECTIVE_LENGTH, :]
    trans_mat = trans[0:EFFECTIVE_LENGTH, :]
    del recei, trans

    # %% range doppler process
    matched_code = load_matched_code(code_file=match_code, verbose=False)
    # matched_code = trans_mat
    # shifted = False
    fft_zoom = 1
    downsampling_rate = 1

    # Conventional Process
    range_data1 = fast_time_correlation(recei_rec,
                                        matched_code=trans_mat,
                                        downsampling_rate=downsampling_rate,
                                        conv_mode='same')
    doppler_data1 = slow_time_fft(range_data1,
                                  win_func=win_func,
                                  shifted=True,
                                  fft_zoom=fft_zoom)

    range_domain = [1500, 15500]
    range_doppler_show(doppler_data1,
                       start_plot=range_domain[0],
                       end_plot=range_domain[1],
                       normalize=False,
                       clim=60,
                       title="{}_range_doppler".format(target),
                       save_fig=save_fig,
                       downsampling_rate=downsampling_rate
                       )

    # %% Doppler compensation
    # doppler_data2 = slow_time_fft(recei_rec,
    #                               win_func=win_func,
    #                               fft_zoom=fft_zoom,
    #                               shifted=False)
    # compensation_data2 = doppler_compensation(doppler_data2,
    #                                           downsampling_rate=downsampling_rate)
    # matched_code = slow_time_fft(trans_mat,
    #                              win_func=win_func,
    #                              fft_zoom=fft_zoom,
    #                              shifted=False)
    #
    # range_data2 = fast_time_correlation(compensation_data2,
    #                                     matched_code=matched_code,
    #                                     downsampling_rate=downsampling_rate)
    #
    # range_doppler_show(range_data2,
    #                    start_plot=range_domain[0],
    #                    end_plot=range_domain[1],
    #                    normalize=False,
    #                    clim=60,
    #                    title="{}_range_doppler_compensation".format(target),
    #                    save_fig=save_fig,
    #                    downsampling_rate=downsampling_rate
    #                    )

    # %%
    # range_doppler_show(20*log10(abs(doppler_data1) + 1e-20)-20*log10(abs(range_data2) + 1e-20),
    #                    start_plot=range_domain[0],
    #                    end_plot=range_domain[1],
    #                    dB=False,
    #                    normalize=False,
    #                    clim=90)
    plt.show()
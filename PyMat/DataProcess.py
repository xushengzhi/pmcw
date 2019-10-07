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

# %% Pre-Setting
save_fig = False
CMAP = 'jet'
window_func = 'hann'
matched = True

if matched:
    match_code = 'pmcw_waveform.txt'
else:
    match_code = 'pmcw_waveform_code2.txt'

# %% Load data
# data = loadmat('data.mat')
# recei = data['data']
# trans = data['tdata']
path = '/Users/shengzhixu/Documents/ExperimentalData/pmcw/cars/'
file ='HH_20190919134453.bin'
target = 'cars'

# path = '/Users/shengzhixu/Documents/ExperimentalData/pmcw/chimney/'
# file ='HH_20190919075847.bin'
# target = 'chimney'

N_Block = 65
recei, trans = data_reform(path + file, n_block_to_process=N_Block, verbose=False, filter=True)

fast_time, slow_time = recei.shape
EFFECTIVE_LENGTH = 262150               # Be careful, this number is obtained by observing the transmitted codes
recei_rec = recei[0:EFFECTIVE_LENGTH, :]
trans_mat = trans[0:EFFECTIVE_LENGTH, :]
del recei, trans


# %% Load match code
def load_matched_code(code_file):
    f = np.linspace(-0.5, 0.5, EFFECTIVE_LENGTH, endpoint=False) * SAMPLING_FREQUENCY
    demix_wave = exp(-2j * pi * CENTRE_FREQUENCY * np.arange(EFFECTIVE_LENGTH) / SAMPLING_FREQUENCY)
    zeroing_length = int(EFFECTIVE_LENGTH / 16)
    wave = np.loadtxt(code_file, delimiter=',')[0::3, 0]
    wave_fft = fftshift(fft(wave[0:EFFECTIVE_LENGTH] * demix_wave))
    wave_fft[0:7 * zeroing_length] = 0
    wave_fft[9 * zeroing_length::] = 0
    w = ifft(ifftshift(wave_fft))
    return w


# %% fast-time correlation
def fast_time_correlation(data, matched_code):
    print("\nFast-time correlation started ...")
    range_data = np.zeros_like(data)
    for i in tqdm(range(data.shape[1])):
        if matched_code.ndim == 1:
            range_data[:, i] = convolve(data[:, i], matched_code[::-1], mode='same')
        else:
            range_data[:, i] = convolve(data[:, i], matched_code[::-1, i], mode='same')

    print("Fast-time correlation finished!")

    return range_data


# %% slow-time FFT
def slow_time_fft(data, window_func=None, fft_zoom=1, shifted=False):
    print("\nSlow-time FFT started ...")
    win = np.ones_like(data)
    if window_func is not None:
        try:
            win_func = eval('sp.signal.windows.' + window_func)
            print("The windowing function {} is applied for slow time".format(win_func.__name__))
            win = np.ones((EFFECTIVE_LENGTH, 1)).dot(win_func(data.shape[1]).reshape(1, data.shape[1]))
        except:
            print('Please provide a valid windowing function: e.g. window_func="hann"')
    doppler_data = fft(data * win, axis=-1, n=data.shape[1]*fft_zoom)
    if shifted:
        doppler_data = fftshift(doppler_data, axes=-1)
    print("Slow-time FFT finished!")
    return doppler_data


# %% images process
def range_doppler_show(data,
                       start_plot=0,
                       end_plot=8000,
                       clim=40,
                       dB = True,
                       normalize=True,
                       title="Range_Doppler",
                       save_fig=False):
    print("\nRange-Doppler imaging started ...")
    dr = C / 2 / SAMPLING_FREQUENCY
    x = np.arange(EFFECTIVE_LENGTH) * dr
    vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION
    v = np.linspace(-vm, vm, data.shape[1], endpoint=False)
    plt.figure()
    if dB:
        data_db = 20 * log10(abs(np.flipud(data[start_plot:end_plot, :])))
    else:
        data_db = np.flipud(data[start_plot:end_plot, :])

    max_db = np.max(data_db)
    print("Maximum amplitude: {:.4}dB".format(max_db))
    if normalize:
        data_db = data_db - max_db
        max_db = 0
    plt.imshow(data_db,
               aspect='auto',
               cmap=CMAP,
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
def doppler_compensation(data, fft_zoom=1):
    print("\nDoppler shifts compensation started ...")
    vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION
    doppler_cells = data.shape[1]*fft_zoom
    v_vector = np.linspace(-vm, vm, doppler_cells, endpoint=False).reshape(1, doppler_cells)
    print("Maximum Doppler shifts is {}Hz".format(1/2/PERIOD_DURATION))
    omega = -2j*pi*2/C*CARRIER_FREQUENCY/SAMPLING_FREQUENCY
    compensation_matrix = exp(omega * np.arange(EFFECTIVE_LENGTH)[::].reshape(EFFECTIVE_LENGTH, 1).dot(v_vector))
    print("Doppler shifts compensation finished!")

    return data * compensation_matrix


# %% range doppler process
matched_code = load_matched_code(code_file=match_code)
fft_zoom = 1

# Conventional Process
range_data1 = fast_time_correlation(recei_rec, matched_code=matched_code)
doppler_data1 = slow_time_fft(range_data1, window_func=window_func, shifted=False, fft_zoom=fft_zoom)

# Doppler compensation
doppler_data2 = slow_time_fft(recei_rec, window_func=window_func, fft_zoom=fft_zoom)
compensation_data2 = doppler_compensation(doppler_data2)
range_data2 = fast_time_correlation(compensation_data2, matched_code=matched_code)

# %% plot
range_domain = [1000, 10000]
range_doppler_show(doppler_data1,
                   start_plot=range_domain[0],
                   end_plot=range_domain[1],
                   normalize=False,
                   clim=60,
                   title="{}_range_doppler".format(target),
                   save_fig=save_fig
                   )

range_doppler_show(range_data2,
                   start_plot=range_domain[0],
                   end_plot=range_domain[1],
                   normalize=False,
                   clim=60,
                   title="{}_range_doppler_compensation".format(target),
                   save_fig=save_fig
                   )

# %%
range_doppler_show(20*log10(abs(doppler_data1))-20*log10(abs(range_data2)),
                   start_plot=range_domain[0],
                   end_plot=range_domain[1],
                   dB=False,
                   normalize=False,
                   clim=30)
plt.show()
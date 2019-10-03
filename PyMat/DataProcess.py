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
file ='HH_20190919134639.bin'

# path = '/Users/shengzhixu/Documents/ExperimentalData/pmcw/chimney/'
# file ='HH_20190919075847.bin'

N_Block = 129
recei, trans = data_reform(path + file, n_block_to_process=N_Block)

fast_time, slow_time = recei.shape
EFFECTIVE_LENGTH = 262150                       # This number is obtained by observing the transmitted codes


# %% low pass filter for matched code
f = np.linspace(-0.5, 0.5, EFFECTIVE_LENGTH)*SAMPLING_FREQUENCY
demix_wave = exp(-2j*pi*CENTRE_FREQUENCY*np.arange(EFFECTIVE_LENGTH)/SAMPLING_FREQUENCY)
zeroing_length = int(EFFECTIVE_LENGTH/8)
wave = np.loadtxt(match_code, delimiter=',')[0::3, 0]
wave_fft = fftshift(fft(wave[0:EFFECTIVE_LENGTH]*demix_wave, axis=0), axes=0)
wave_fft[0:3*zeroing_length] = 0
wave_fft[5*zeroing_length::] = 0
w = ifft(ifftshift(wave_fft, axes=0), axis=0)


# %% fast-time correlation
recei_rec = recei[0:EFFECTIVE_LENGTH, :]


def fast_time_correlation(data, matched_code):
    print("\nFast-time correlation started ...")
    range_data = np.zeros_like(data)
    for i in tqdm(range(slow_time)):
        range_data[:, i] = convolve(data[:, i], matched_code[::-1], mode='same')
    print("Fast-time correlation finished!")

    return range_data


# %% slow-time FFT
def slow_time_fft(data, window_func=None):
    print("\nSlow-time FFT started ...")
    win = np.ones_like(data)
    if window_func is not None:
        try:
            win_func = eval('sp.signal.windows.' + window_func)
            print("The windowing function {} is applied for slow time".format(win_func.__name__))
            win = np.ones((EFFECTIVE_LENGTH, 1)).dot(win_func(slow_time).reshape(1, slow_time))
        except:
            print('Please provide a valid windowing function: e.g. window_func="hann"')
    doppler_data = fft(data * win, axis=-1, n=128)
    print("Slow-time FFT finished!")
    return doppler_data


# %% images process
def range_doppler_show(data, end_plot=8000):
    print("\nRange-Doppler imaging started ...")
    dr = C / 2 / SAMPLING_FREQUENCY
    x = np.arange(EFFECTIVE_LENGTH) * dr
    vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION
    plt.figure()
    plt.imshow(20 * log10(abs(np.flipud(data[0:end_plot, :])) + 1e-20),
               aspect='auto',
               cmap=CMAP,
               extent=[-vm, vm, x[0], x[end_plot]])
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Range (m)')
    cbar = plt.colorbar()
    cbar.set_label("(dB)")
    print("\nRange-Doppler imaging Finished")


# %% Doppler compensation
def doppler_compensation(data):
    pass







# %% Conventional Process
range_data = fast_time_correlation(recei_rec, matched_code=w)
doppler_data = slow_time_fft(range_data, window_func=window_func)
range_doppler_show(doppler_data)

# %% Doppler compensation


# %%
plt.show()
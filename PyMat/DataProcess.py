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

# %% Pre-Setting
save_fig = False
CMAP = 'jet'
window_func = 'hann'
matched = False

if matched:
    match_code = 'pmcw_waveform.txt'
else:
    match_code = 'pmcw_waveform_code2.txt'


# %% System paras
fs = 400e6
fc = 125e6
Fc = 3.315e9
B = 50e6
T = 1e-3

# %% Load data
# data = loadmat('data.mat')
# recei = data['data']
# trans = data['tdata']
path = '/Users/shengzhixu/Documents/ExperimentalData/pmcw/cars/'
file ='HH_20190919134639.bin'

# path = '/Users/shengzhixu/Documents/ExperimentalData/pmcw/chimney/'
# file ='HH_20190919075847.bin'


recei, trans = data_reform(path + file, n_block_to_process=129)

fast_time, slow_time = recei.shape
effective_length = 262150

plt.figure()
plt.imshow(trans[0:effective_length, :], cmap=CMAP, aspect='auto')

# recei_fft = fftshift(fft(recei, axis=-1, n=128), axes=-1)
# plt.figure()
# plt.imshow(abs(recei_fft), cmap=CMAP, aspect='auto')

# %% low pass filter
recei_eff = recei[0:effective_length, :]
f = np.linspace(-0.5, 0.5, effective_length)*fs

demix_wave = exp(-2j*pi*fc*np.arange(effective_length)/fs)
demw = demix_wave.reshape(effective_length, 1).dot(np.ones((1, slow_time)))
recei_fft = fftshift(fft(recei_eff*demw, axis=0), axes=0)

zeroing_length = int(effective_length/4)
recei_fft[0:zeroing_length, :] = 0
recei_fft[3*zeroing_length::, :] = 0

plt.figure()
plt.plot(f, 20*log10(abs(recei_fft[:, 0] * demix_wave) + 1e-20))

recei_rec = ifft(ifftshift(recei_fft, axes=0), axis=0)

plt.figure()
plt.plot(f, np.real(recei_rec[:, 16]))

# %% matched code
wave = np.loadtxt(match_code, delimiter=',')[0::3, 0]
wave_fft = fftshift(fft(wave[0:effective_length]*demix_wave, axis=0), axes=0)
wave_fft[0:zeroing_length] = 0
wave_fft[3*zeroing_length::] = 0

plt.figure()
plt.plot(f, 20*log10(abs(wave_fft) + 1e-20))
w = ifft(ifftshift(wave_fft, axes=0), axis=0)

# %% fast-time correlation
CORR = np.zeros_like(recei_rec)
for i in tqdm(range(slow_time)):
    CORR[:, i] = convolve(recei_rec[:, i], w[::-1], mode='same')

plt.figure()
plt.imshow(20*log10(abs(CORR) + 1e-20), aspect='auto', cmap=CMAP)


# %% slow-time FFT
win = np.ones_like(CORR)
if window_func is not None:
    try:
        win_func = eval('sp.signal.windows.'+window_func)
        print("the windowing function {} is applied for slow time".format(win_func.__name__))
        win = np.ones((effective_length, 1)).dot(win_func(slow_time).reshape(1, slow_time))
    except:
        print('Please provide a valid windowing function: e.g. window_func="hann"')

CORRD = fftshift(fft(CORR*win, axis=-1, n=128), axes=-1)

# %% images process
dr = C/2/fs
x = np.arange(effective_length)*dr
vm = C/4/Fc/T

end_plot = 12000
plt.figure()
plt.imshow(20*log10(abs(np.flipud(CORRD[0:end_plot, :])) + 1e-20),
           aspect='auto',
           cmap=CMAP,
           extent=[-vm, vm, x[0], x[end_plot]])
plt.xlabel('Velocity (m/s)')
plt.ylabel('Range (m)')


# %%
plt.show()
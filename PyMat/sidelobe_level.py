# -*- codeing: utf-8 -*-
'''
Create on 2020/2/3

@author: shengzhixu
@email:sz.xu@hotmial.com

Simulations according to exact same parameters with exponential data
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.constants import speed_of_light as C
from numpy import exp, pi
from tqdm import tqdm
from scipy.signal import convolve
from numpy.fft import fft, fftshift, ifft

from SysParas import PERIOD_DURATION, CARRIER_FREQUENCY, SAMPLING_FREQUENCY, BANDWIDTH, EFFECTIVE_LENGTH
from utils import dB, roll_zeropad


# %% Code Waveform Generator
codes = loadmat('code8192.mat')['codes']
code1 = codes[0, :]
code2 = codes[1, :]
code_length = code1.size
code_repeat = 16

code1 = np.kron(code1, np.ones((code_repeat, )))
code2 = np.kron(code2, np.ones((code_repeat, )))
code1_repeat = code1
code2_repeat = code2
total_zero_padding = 200_000
total_zero_padding = EFFECTIVE_LENGTH
code1 = np.pad(code1, (0, total_zero_padding-EFFECTIVE_LENGTH), 'constant', constant_values=(0, 0))
code2 = np.pad(code2, (0, total_zero_padding-EFFECTIVE_LENGTH), 'constant', constant_values=(0, 0))
code1_pad = code1
code2_pad = code2

plt.figure()
plt.step(np.arange(code1.size), code1)
print("wave size of each period: ", code1.size)

period_repeat = 32
total_length = code1.size*period_repeat

code1 = np.kron(np.ones((period_repeat, )), code1)
code2 = np.kron(np.ones((period_repeat, )), code2)


# %% Filtered
code1_fft = fft(code1_pad)
tail_length = code1_fft.size//8
code1_fft[tail_length:7*tail_length] = 0
code1 = ifft(code1_fft)[0:EFFECTIVE_LENGTH]
tail_length = code2.size//8
code2_fft = fft(code2_pad)
code2_fft[tail_length:7*tail_length] = 0
code2 = ifft(code2_fft)[0:EFFECTIVE_LENGTH]


# %% Target Setting
mycorr = convolve

Tc = 2/BANDWIDTH
vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION

R = 10            # meters
resolution = C/2/BANDWIDTH
shifted = int(R/resolution*code_repeat)
omega = 2j*pi*2/C*CARRIER_FREQUENCY/SAMPLING_FREQUENCY
lr_aixs = EFFECTIVE_LENGTH//2

# v_test_vector = np.linspace(0, vm, 10, endpoint=False)
v_test_vector= [-vm]

max_acf_dB = np.zeros_like(v_test_vector)
max_acf_com_dB = np.zeros_like(v_test_vector)
max_ccf_dB = np.zeros_like(v_test_vector)
max_ccf_com_dB = np.zeros_like(v_test_vector)

for j,v in tqdm(enumerate(v_test_vector)):

    # %% filtering received signal
    trans_code = np.kron(np.ones((period_repeat,)), code1_pad)
    receive_code1 = roll_zeropad(trans_code, shifted) * exp(omega * v * np.arange(total_length)) \
                    + (np.random.randn(total_length) + 1j*np.random.randn(total_length)) * 0
    receive_code1_fft = fft(receive_code1)
    receive_code1_fft[tail_length:7*tail_length] = 0
    receive_code1_filtered = ifft(receive_code1_fft)
    receive_code1_filtered = receive_code1_filtered.reshape(period_repeat, total_zero_padding)
    receive_code1_filtered = receive_code1_filtered[:, 0:EFFECTIVE_LENGTH]

    # %% correlation and FFT
    acf = np.zeros_like(receive_code1_filtered)
    acf_com = np.zeros_like(receive_code1_filtered)
    ccf = np.zeros_like(receive_code1_filtered)
    ccf_com = np.zeros_like(receive_code1_filtered)

    win_func = np.kron(np.ones((EFFECTIVE_LENGTH,)), sp.signal.windows.hann(period_repeat))\
        .reshape(EFFECTIVE_LENGTH, period_repeat).T

    # fs
    for i in tqdm(range(period_repeat)):
        acf[i, :] = mycorr(receive_code1_filtered[i, :], code1[::-1].conj(), mode='same')
        ccf[i, :] = mycorr(receive_code1_filtered[i, :], code2[::-1].conj(), mode='same')
    acf = fftshift(fft(acf*win_func, axis=0), axes=0)
    ccf = fftshift(fft(ccf*win_func, axis=0), axes=0)


    # sf with compensation
    receive_filter_fft = fftshift(fft(receive_code1_filtered*win_func, axis=0), axes=0) # (-vm, vm)
    v_vector = np.linspace(-vm, vm, period_repeat, endpoint=False).reshape(period_repeat, 1)
    compensation_matrix = exp(-omega * v_vector @ \
                              np.arange(EFFECTIVE_LENGTH).reshape(1, EFFECTIVE_LENGTH)) # (-vm, vm)
    receive_code1_filtered_com = receive_filter_fft * compensation_matrix

    for i in tqdm(range(period_repeat)):
        acf_com[i, :] = (mycorr(receive_code1_filtered_com[i, :], code1[::-1].conj(), mode='same'))
        ccf_com[i, :] = mycorr(receive_code1_filtered_com[i, :], code2[::-1].conj(), mode='same')

    acf_dB = dB(acf).T
    ccf_dB = dB(ccf).T
    acf_com_dB = dB(acf_com).T
    ccf_com_dB = dB(ccf_com).T

    max_acf_dB[j] = np.max(acf_dB)
    max_acf_com_dB[j] = np.max(acf_com_dB)
    max_ccf_dB[j] = np.max(ccf_dB)
    max_ccf_com_dB[j] = np.max(ccf_com_dB)

# %% ssl
plt.figure()
plt.plot(v_test_vector, max_acf_dB, label='acf')
plt.plot(v_test_vector, max_acf_com_dB, label='acf_com')
plt.legend(loc='upper right')

print((max_acf_com_dB - max_acf_dB))

plt.figure()
plt.plot(v_test_vector, max_ccf_dB, label='ccf')
plt.plot(v_test_vector, max_ccf_com_dB, label='ccf_com')
plt.legend(loc='upper right')

# %% show results

# print('Max acf:', np.max(acf_dB))
# print('Max ccf:', np.max(ccf_dB))
# print('Max acf_com:', np.max(acf_com_dB))
# print('Max ccf_com:', np.max(ccf_com_dB))
#
# print('ACF increment: ', np.max(acf_com_dB) - np.max(acf_dB), 'dB')
#
# clim = 60
# plt.figure()
# plt.imshow(acf_dB[EFFECTIVE_LENGTH//2:EFFECTIVE_LENGTH//2+2000, :], cmap='jet', aspect='auto')
# plt.clim([np.max(acf_dB)-clim, np.max(acf_dB)])
# plt.title('acf')
# plt.colorbar()
#
# # plt.figure()
# # plt.imshow((ccf_dB), cmap='jet', aspect='auto')
# # plt.clim([np.max(ccf_dB)-50, np.max(ccf_dB)])
# # plt.title('ccf')
# # plt.colorbar()
#
# plt.figure()
# plt.imshow(acf_com_dB[EFFECTIVE_LENGTH//2:EFFECTIVE_LENGTH//2+2000, :], cmap='jet', aspect='auto')
# plt.clim([np.max(acf_com_dB)-clim, np.max(acf_com_dB)])
# plt.title('acf_com')
# plt.colorbar()
#
# # plt.figure()
# # plt.imshow((ccf_com_dB), cmap='jet', aspect='auto')
# # plt.clim([np.max(ccf_com_dB)-50, np.max(ccf_com_dB)])
# # plt.title('ccf_com')
# # plt.colorbar()
#
# # %%
# plt.figure()
# x_axis = np.arange(-EFFECTIVE_LENGTH//2, EFFECTIVE_LENGTH//2)
# plt.plot(x_axis, acf_com_dB[:, 8], alpha=0.8, label='acf_com')
# plt.plot(x_axis, acf_dB[:, 8], alpha=0.8, label='acf')
# plt.grid(ls='-.')
# plt.legend(loc='upper right')
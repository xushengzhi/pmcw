# -*- coding: utf-8 -*-
'''
Creat on 27/01/2020

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fftpack import fft, fftshift, ifft, ifftshift
from scipy.signal import convolve, correlate
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from utils import dB, conv_circ


# %%
save_fig = False


# %%
matched_wave = 'pmcw_8192.txt'
mismatched_wave = 'pmcw_8192_miscode.txt'

matched_intermediate_wave = np.loadtxt('pmcw_8192.txt', delimiter=',')[:, 0][0:131071]
mismatched_intermediate_wave = np.loadtxt('pmcw_8192_miscode.txt', delimiter=',')[:, 0][0:131071]

code = loadmat('code8192.mat')['codes']
repeat_times = 16
matched_code = np.kron(code[0, :], np.ones(repeat_times, ))
mismatched_code = np.kron(code[1, :], np.ones(repeat_times, ))


# %% original code
# TODO: Incorrect codes for 1.2GHz (we do not have this, but )
# print("Code length (with repetition of {}): {}".format(repeat_times, matched_code.size))
# acf400 = dB(np.convolve(matched_code, matched_code[::-1], mode='same'))
# # acf1200 = conv_circ(matched_code[::1], matched_code[::-1])
# plt.figure()
# plt.plot(acf400)
# plt.title("1200MHz")
# plt.ylim([0, 105])
#
# ind_max = np.argmax(acf400)
# max_val = acf400[ind_max]
# acf400[ind_max-100:ind_max+100] = np.min(acf400)
# sidelobe_max = np.max(acf400)
#
# print("PSLR is {}dB - {}dB = {}dB".format(max_val, sidelobe_max, max_val - sidelobe_max))


# %% down-sample at fs = 400MHz
# code400M = np.zeros((400_000, ))
# code400M[0:8192*16] = matched_code[::1]
code400M = matched_code
acf400 = dB(convolve(code400M, code400M[::-1].conj(), mode='same', method='fft'))
# acf400 = conv_circ(matched_code[::3], matched_code[::-3])
plt.figure()
plt.plot((acf400))
plt.title("400MHz")
plt.ylim([0, 105])

acf400_copy = acf400.copy()
ind_max = np.argmax(acf400_copy)
max_val = acf400_copy[ind_max]
acf400_copy[ind_max-100:ind_max+100] = np.min(acf400_copy)
sidelobe_max = np.max(acf400_copy)

print("PSLR is {}dB - {}dB = {}dB".format(max_val, sidelobe_max, max_val - sidelobe_max))

# %% low-pass filter at B = 50MHz
matched_code_downsampled = matched_code[::1]
matched_code_fft = fft(code400M)
plt.figure(figsize=[8, 4])
plt.plot(np.linspace(-200, 200, code400M.size, endpoint=False), dB(np.fft.fftshift((matched_code_fft))))
plt.ylim([0, 40])
plt.grid(ls=':')

plt.xlabel('Frequency (MHz)')
plt.ylabel('Spectrum (dB)')

filter100M = np.zeros(400)
filter100M[150:250] = 38
plt.plot(np.linspace(-200, 200, 400, endpoint=False), filter100M, c='g', lw=2, label='100MHz Filter')

filter50M = np.zeros(400)
filter50M[175:225] = 37.5
plt.plot(np.linspace(-200, 200, 400, endpoint=False), filter50M, c='r', lw=2, label='50MHz Filter')
plt.legend()
plt.tight_layout()
if save_fig:
    plt.savefig('theoretical_filter.png', dpi=300)



# %%
tail_length = code400M.size//16
matched_code_fft[2*tail_length:14*tail_length] = 0
plt.figure()
plt.plot(abs(fftshift(matched_code_fft)))

code100M = ifft(matched_code_fft)

matched_code_fft[1*tail_length:15*tail_length] = 0
code50M = ifft(matched_code_fft)

# %% plot
acf50 = dB(convolve(code50M, code50M[::-1].conj(), mode='same'))
acf100 = dB(convolve(code100M, code100M[::-1].conj(), mode='same'))
acf50_copy = acf50.copy()
acf100_copy = acf100.copy()
ind_max = np.argmax(acf50_copy)
max_val = acf50_copy[ind_max]
acf50_copy[ind_max-100:ind_max+100] = np.min(acf50_copy)
sidelobe_max = np.max(acf50_copy)
norm_value = max(np.max(acf50), np.max(acf400))

# %%
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(-65536, 65536), acf400-norm_value, label="400MHz", alpha=1, lw=2)
ax.plot(np.arange(-65536, 65536), acf100-norm_value, label="100MHz", alpha=1, lw=2, c='g')
ax.plot(np.arange(-65536, 65536), acf50-norm_value, label="50MHz", alpha=1, lw=2, c='r')
axins = zoomed_inset_axes(ax, 3.5, loc=2)
axins.plot(np.arange(-65536, 65536), acf400-norm_value, label="400MHz", alpha=1, lw=2)
axins.plot(np.arange(-65536, 65536), acf100-norm_value, label="100MHz", alpha=1, lw=2, c='g')
axins.plot(np.arange(-65536, 65536), acf50-norm_value, label="50MHz", alpha=1, lw=2, c='r')
axins.set_xlim([-10, 10])
axins.set_ylim([-5, 3])
axins.grid(ls=':')
plt.xticks(visible=False)
axins.yaxis.tick_right()
# plt.yticks(visible=False)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.set_ylim([-60, 5])
ax.set_xlim([-150, 150])
ax.legend(loc='upper right')
ax.set_xlabel('Index lag')
ax.set_ylabel('Autocorrelation (dB)')
plt.tight_layout()
ax.grid(axis='y', linestyle=':')
if save_fig:
    plt.savefig('limit_bandwidth.png', dpi=300)
# plt.title("50MHz")

print("PSLR is {}dB - {}dB = {}dB".format(max_val, sidelobe_max, max_val - sidelobe_max))
# %%
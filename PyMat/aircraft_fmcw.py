# -*- codeing: utf-8 -*-
'''
Create on 2020/2/11

@author: shengzhixu
@email:sz.xu@hotmial.com
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.constants import speed_of_light as C
from scipy.signal import stft
from numpy.fft import fft, fftshift, fft2

from DataReform import data_reform
from SysParas import CARRIER_FREQUENCY, BANDWIDTH
from utils import load_matched_code, dB
from DataProcess import fast_time_correlation, slow_time_fft, doppler_compensation

# %% load data
path = '/Volumes/Personal/ExternalDrive/Backup/PMCWPARSAXData/aircraft/'
# file = 'VV_20200207111141.bin'
file = 'HV_20200207111148.bin'

PERIOD_DURATION = 1e-4
save_fig = False
INTERMEDIATE_FREQUENCY = 125_000_000
SAMPLING_FREQUENCY = 399_996_327
EFFECTIVE_LENGTH = 2048*16
n_block_to_process = 45
fft_zoom = 2

recei, trans = data_reform(path + file,
                          verbose=False,
                          filter=False,
                          compensation=False,
                          n_block_to_process=n_block_to_process,
                          win_func='rect',
                          fi=INTERMEDIATE_FREQUENCY,
                          pri=PERIOD_DURATION,
                          fisrt_sidelobe_include=True)

# %% stft show
fasttime, slowtime = trans.shape

# f, t, Zxx = stft(recei[:, 5].real, fs=1, nperseg=256, noverlap=255)
#
# plt.figure()
# plt.imshow(dB(Zxx[0:40, :]))
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.title('Recei')
# plt.xlabel('time series')
# plt.ylabel('frequency')
#
# f, t, Zxx = stft(trans[:, 5].real, fs=1, nperseg=256, noverlap=255)
#
# plt.figure()
# plt.imshow(dB(Zxx[0:40, :]))
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.xlabel('time series')
# plt.ylabel('frequency')
# plt.title('Trans')


# %% load matched_code
# matched_code = load_matched_code('data/fmcw2048.txt', fi=INTERMEDIATE_FREQUENCY, verbose=True)
# f, t, Zxx = stft(matched_code.real, fs=1, nperseg=256, noverlap=255)
#
# plt.figure()
# plt.imshow(dB(Zxx[0:40, :]))
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.title('Matched')
# matching_code =  matched_code.reshape(fasttime, 1) @ np.ones((1, slowtime))


# %% digitally de-chirping
dechirping = recei * trans.conj()

# f, t, Zxx = stft(dechirping[:, 0].real, fs=1, nperseg=1024, noverlap=1023)
#
# plt.figure()
# plt.imshow(dB(Zxx[0:200, :]))
# plt.gca().invert_yaxis()
# plt.colorbar()

# %%
rd = fftshift(fft2(dechirping, [fasttime*fft_zoom, slowtime*fft_zoom] ))
# %%
vm = C/4/PERIOD_DURATION/CARRIER_FREQUENCY
resolution = C/2/BANDWIDTH
range_domain = np.arange(fasttime*fft_zoom//4) * resolution /fft_zoom/2
velocity_domain = np.linspace(-vm, vm, slowtime*fft_zoom, endpoint=False)

fig, ax = plt.subplots()
img = ax.imshow(dB(rd[fasttime*fft_zoom//4:fasttime*fft_zoom//2, :]),
           extent=[velocity_domain[0], velocity_domain[-1], range_domain[0]/1000, range_domain[-1]/1000])
# plt.gca().invert_yaxis()
plt.colorbar(img)
img.set_clim([30, 100])
ax.set_xlabel('velocty (m/s)')
ax.set_ylabel('Range (km)')
matched = True

if matched:
    axins = inset_axes(ax, width="30%", height="30%", loc=1)
    imgin = axins.imshow(dB(rd[fasttime * fft_zoom // 4:fasttime * fft_zoom // 2, :]),
              extent=[velocity_domain[0], velocity_domain[-1], range_domain[0] / 1000, range_domain[-1] / 1000])
    # plt.colorbar(imgin)
    imgin.set_clim([30, 100])
    axins.set_xlim([200, 220])
    axins.set_ylim([1.255, 1.230])
    # plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")



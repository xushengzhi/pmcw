# -*- codeing: utf-8 -*-
'''
Create on 2020/3/3

@author: shengzhixu
@email:sz.xu@hotmial.com
'''

import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.constants import speed_of_light as C
from numpy import sqrt, pi, cos

from DataReform import data_reform
from SysParas import CARRIER_FREQUENCY, BANDWIDTH
from utils import load_matched_code, dB
from DataProcess import fast_time_correlation, slow_time_fft, doppler_compensation

# path = '/Volumes/Personal/ExternalDrive/Backup/PMCWPARSAXData/schonmarkweg/'
# file = 'HH_20200303095930.bin'
# pri = 1e-3 / 2
# save_fig = False

path = "/Users/shengzhixu/Documents/Data/"
file = "VV_20200309103107.bin"


# path =  '/Volumes/Personal/Backup/PMCWPARSAXData/A13/'
# channel = 'VV'
# file = channel + '_20191009081112.bin'
# pri = 0.5e-3

INTERMEDIATE_FREQUENCY = 125_000_000
SAMPLING_FREQUENCY = 399_996_327
# SAMPLING_FREQUENCY = 400e6
PERIOD_DURATION = 1.311e-3

recei, trans = data_reform(path + file,
                          verbose=True,
                          filter=True,
                          n_block_to_process=201,
                          win_func='rect',
                          pri=PERIOD_DURATION)

EFFECTIVE_LENGTH, _ = trans.shape
fft_zoom = 1
# %%
matched = True
if matched:
    match_code = 'data/pmcw_32768_zeropad_False.txt'

matched_code = load_matched_code(code_file=match_code,
                                 verbose=False,
                                 win_func='rect',
                                 compensated=True,
                                 fisrt_sidelobe_include=True)
vm = C/ 4 / CARRIER_FREQUENCY / PERIOD_DURATION

fasttime, slowtime = trans.shape
resolution = C/2/BANDWIDTH
range_domain = np.arange(EFFECTIVE_LENGTH//2) * resolution / 16
velocity_domain = np.linspace(-vm, vm, recei.shape[1]*fft_zoom, endpoint=False)

# %%

# fs
tic = time.time()
range_data = fast_time_correlation(recei[0:EFFECTIVE_LENGTH, :],
                                   matched_code = matched_code,
                                   conv_mode='same')
doppler_data = slow_time_fft(range_data,
                             win_func='hann',
                             fft_zoom=fft_zoom,
                             shifted=True)

# sf
doppler_data2 = slow_time_fft(recei[0:EFFECTIVE_LENGTH, :],
                             win_func='hann',
                             fft_zoom=fft_zoom,
                             shifted=True)
compensation_data = doppler_compensation(doppler_data2, pri=PERIOD_DURATION)
range_data2 = fast_time_correlation(compensation_data,
                                   matched_code=matched_code,
                                   conv_mode='same')
toc = time.time()
print("Processing time: {:.3} seconds".format(toc - tic))
# %%
plt.figure()
plt.imshow(dB(doppler_data[EFFECTIVE_LENGTH//2::, :]))
plt.clim([20, 60])
plt.gca().invert_yaxis()


save_fig = 1
range_axis = [9000, 11500]
plt.figure()
plt.imshow(dB(doppler_data[EFFECTIVE_LENGTH//2+range_axis[0]:EFFECTIVE_LENGTH//2+range_axis[1], :]),
           extent=[velocity_domain[0], velocity_domain[-1],
                   range_domain[range_axis[0]]/1000, range_domain[range_axis[1]]/1000])
plt.xlabel('Velocity (m/s)')
plt.ylabel('Range (km)')
# plt.gca().invert_yaxis()
plt.colorbar()
plt.clim([20, 60])
if save_fig:
    plt.savefig('fs_vehicle.png', dpi=300)

plt.figure()
plt.imshow(dB(range_data2[EFFECTIVE_LENGTH//2+range_axis[0]:EFFECTIVE_LENGTH//2+range_axis[1], :]),
           extent=[velocity_domain[0], velocity_domain[-1],
                   range_domain[range_axis[0]]/1000, range_domain[range_axis[1]]/1000])
plt.xlabel('Velocity (m/s)')
plt.ylabel('Range (km)')
# plt.gca().invert_yaxis()
plt.colorbar()
plt.clim([20, 60])
if save_fig:
    plt.savefig('sf_vehicle.png', dpi=300)


# velocity_slice = 419 # 954//2
# increment = np.max(dB(range_data2[EFFECTIVE_LENGTH//2::, velocity_slice])) - \
#                 np.max(dB(doppler_data[EFFECTIVE_LENGTH//2::, velocity_slice]))
# sigma = 2*pi*(EFFECTIVE_LENGTH-3203)/((PERIOD_DURATION)* SAMPLING_FREQUENCY) * \
#         abs(velocity_domain[velocity_slice]) / vm/2
#
# print('Theoretical increment: ', -dB(sqrt(2-2*cos(sigma))/sigma))
# print('dB increase: ', increment)
#
# matched = 1
#
# fig, ax = plt.subplots(figsize=(15, 4))
# ax.plot(range_domain/1000, dB(range_data2[EFFECTIVE_LENGTH//2::, velocity_slice]), label='with compensation')
# ax.plot(range_domain/1000, dB(doppler_data[EFFECTIVE_LENGTH//2::, velocity_slice]), label='without compensation')
# ax.set_xlabel('Range (km)')
# ax.set_ylabel('(dB)')
# ax.set_ylim([20, 55])
# ax.set_xlim([1.5, 2.5])
# ax.grid(ls=':')
# ax.legend(loc='upper right')
# plt.tight_layout()
# if matched:
#     axins = zoomed_inset_axes(ax, 2.5, loc=2)
#     axins.plot(range_domain/1000, dB(range_data2[EFFECTIVE_LENGTH//2::, velocity_slice]), label='with compensation')
#     axins.plot(range_domain/1000, dB(doppler_data[EFFECTIVE_LENGTH//2::, velocity_slice]), label='without compensation')
#     axins.set_xlim([1.94, 1.98])
#     axins.set_ylim([42, 52])
#     axins.grid(ls=':')
#     plt.xticks(visible=False)
#     axins.yaxis.tick_right()
#     mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# if save_fig:
#     plt.savefig('slicing.png', dpi=300)

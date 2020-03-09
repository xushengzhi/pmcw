# -*- coding: utf-8 -*-
'''
Creat on 30/09/2019

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.io import loadmat
from tqdm import tqdm
from scipy.constants import speed_of_light as C
from scipy.io import savemat, loadmat
from numpy import pi, exp, log10
from numpy.fft import fft, fftshift, ifftshift, ifft
from scipy.signal import convolve

from DataReform import data_reform
from SysParas import *
from utils import load_matched_code, next_pow, conv_fft, deconv_fft, dB


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
            # range_data[:, i] = conv_fft(data_downsampling[:, i], matched_code_downsampling[:, i], mode=conv_mode)[0:range_data.shape[0]]
    print("Fast-time correlation finished!")

    return range_data


def fast_time_decorrelation(data, matched_code):

    print("\nFast-time decorrelation started ...")
    match_code = matched_code[::-1].conj()
    range_data = np.zeros((EFFECTIVE_LENGTH, data.shape[1]), dtype='complex')
    for i in tqdm(range(data.shape[1])):
        range_data[:, i] = deconv_fft(data[:, i], match_code)[0:EFFECTIVE_LENGTH]

    print("\nFast-time decorrelation finished!")

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


# %% images show
def range_doppler_show(data,
                       start_plot=0,
                       end_plot=8000,
                       downsampling_rate=1,
                       clim=40,
                       dbscale = True,
                       normalize=True,
                       cmap='jet',
                       title="Range_Doppler",
                       pic_title=None,
                       save_fig=False,
                       FIGSIZE=None):
    if FIGSIZE is None:
        FIGSIZE = [6, 7]
    print("\nRange-Doppler imaging started ...")
    dr = C / 2 / SAMPLING_FREQUENCY * downsampling_rate
    x = np.arange(EFFECTIVE_LENGTH) * dr
    vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION
    v = np.linspace(-vm, vm, data.shape[1], endpoint=False)
    plt.figure(figsize=FIGSIZE)
    if dbscale:
        data_db = dB(np.flipud(data[start_plot:end_plot, :]))
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
               extent=[v[0], v[-1], (x[start_plot-65536])/1000, (x[end_plot-65536])/1000])
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Range (km)')
    if type(clim) is int:
        plt.clim([max_db-clim, max_db])
    else:
        plt.clim(clim)
    if pic_title is not None:
        plt.title(pic_title)
    cbar = plt.colorbar()
    cbar.set_label("(dB)", rotation=-90, labelpad=14)

    # A = 126.3
    # B = 109.7
    #
    # # A = 132.0
    # # B = 110.0
    #
    # # A = 72.1
    # # B = 105.0
    #
    # # A = 73.1
    # # B = 105.1
    #
    # plt.annotate('{}'.format(A), xy=(29.8, 3.837), xytext=(35.19, 3.755),
    #             arrowprops=dict(facecolor='b', shrink=0.05, width=1.0, headwidth=3.5),
    #              size=12
    #             ) # '${:.1f}$'.format(data_db[739, 332])
    #
    # plt.annotate('{}'.format(B), xy=(29.8, 4.335), xytext=(35.19, 4.250),
    #             arrowprops=dict(facecolor='b', shrink=0.05, width=1.0, headwidth=3.5),
    #              size=12
    #             )

    if save_fig:
        plt.savefig("{}.png".format(title), dpi=300)
    print("Range-Doppler imaging Finished")

    return data_db


# %% Doppler compensation
def doppler_compensation(data, downsampling_rate=1, pri=1e-3):

    print("\nDoppler shifts compensation started ...")
    vm = C / 4 / CARRIER_FREQUENCY / pri
    doppler_cells = data.shape[1]
    v_vector = np.linspace(-vm, vm, doppler_cells, endpoint=False).reshape(1, doppler_cells)

    print("Maximum Doppler shifts is {}Hz".format(1/2/PERIOD_DURATION))
    omega = 2j*pi*2/C*CARRIER_FREQUENCY/SAMPLING_FREQUENCY*downsampling_rate
    compensation_matrix = exp(-omega * np.arange(data.shape[0]).reshape(data.shape[0], 1).dot(v_vector))

    print("Doppler shifts compensation finished!")

    return data * compensation_matrix


if __name__ == '__main__':
    # %% Pre-Setting
    # save_fig = 1
    CMAP = 'jet'
    win_func = 'hann'
    matched = False

    if matched:
        match_code = 'pmcw_8192.txt'
    else:
        match_code = 'pmcw_8192_miscode.txt'

    # %% Load data
    # data = loadmat('data.mat')
    # recei = data['data']
    # trans = data['tdata']
    # path =  '/Volumes/Personal/Backup/PMCWPARSAXData/cars/'
    # file ='HH_20190919134453.bin'

    path = '/Volumes/Personal/ExternalDrive/Backup/PMCWPARSAXData/sync_clock_data_a13/'
    file = 'VV/VV_20191126093918.bin'
    file = 'VV/VV_20191126093852.bin'

    target = 'cars'
    PERIOD_DURATION = 1e-3/2
    vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION

    # path = '/Volumes/Personal/Backup/PMCWPARSAXData/chimney/'
    # file ='HH_20190919075858.bin'
    # target = 'chimney'

    # path =  '/Volumes/Personal/Backup/PMCWPARSAXData/A13/'
    # file = 'VV_20191009081112.bin'
    # target = 'vehicles'
    # PERIOD_DURATION = 0.5e-3
    # %%
    N_Block = 201
    fft_zoom = 2
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

    # %% pre-setting
    matched_code = load_matched_code(code_file=match_code, verbose=False, win_func='rect', compensated=True)
    downsampling_rate = 1

    # %% Conventional process
    # fast - slow
    range_data3 = fast_time_correlation(recei_rec,
                                        matched_code=matched_code,
                                        downsampling_rate=downsampling_rate,
                                        conv_mode='same')

    doppler_data3 = slow_time_fft(range_data3,
                                  win_func=win_func,
                                  fft_zoom=fft_zoom,
                                  shifted=True)

    #  slow fast
    doppler_data4 = slow_time_fft(recei_rec,
                                  win_func=win_func,
                                  fft_zoom=fft_zoom,
                                  shifted=True)

    compensation_data4 = doppler_compensation(doppler_data4, pri=PERIOD_DURATION)

    range_data4 = fast_time_correlation(compensation_data4,
                                        matched_code=matched_code,
                                        downsampling_rate=downsampling_rate,
                                        conv_mode='same')

#%% plot


    range_domain = np.array([9500, 12000]) + 65536
    clim = [35, 65]
    CMAP = 'hot_r'
    save_fig = 1 # please always keep it as False

    FIGSIZE = [7, 6]
    print('Range: ', np.array(range_domain)*0.375)
    dd_db3 = range_doppler_show(doppler_data3,
                               start_plot=range_domain[0],
                               end_plot=range_domain[1],
                               normalize=False,
                               clim=clim,
                               title="{}_range_doppler_fs_match_{}".format(target, str(matched)),
                               save_fig=save_fig,
                               downsampling_rate=downsampling_rate,
                               FIGSIZE=FIGSIZE,
                                cmap=CMAP
                       )

    rd_db4 = range_doppler_show(range_data4,
                               start_plot=range_domain[0],
                               end_plot=range_domain[1],
                               normalize=False,
                               clim=clim,
                               title="{}_range_doppler_sf_match_{}".format(target, str(matched)),
                               save_fig=save_fig,
                               downsampling_rate=downsampling_rate,
                                FIGSIZE=FIGSIZE,
                                cmap=CMAP
                       )

# %% sliced Doppler data for better comparison

    v = np.linspace(-vm, vm, rd_db4.shape[1], endpoint=False)
    # slice_velocity = 28   # 75, 55, 332
    dv = v[1] - v[0]
    v_slice = -32.68
    slice_velocity = int((v_slice + vm)/dv)
    print("velocity resolution: {} m/s".format(v[2] - v[1]))
    print("Plot sliced data at velocity: {}".format(v[slice_velocity]))


    # plt.figure(figsize=[15, 4])
    # plt.plot((np.arange(range_domain[0], range_domain[1])-EFFECTIVE_LENGTH//2)*0.375/1000,
    #          rd_db4[::-1, slice_velocity], label='with compensation')
    # plt.plot((np.arange(range_domain[0], range_domain[1])-EFFECTIVE_LENGTH//2)*0.375/1000,
    #          dd_db3[::-1, slice_velocity], label='without compensation')
    # plt.grid(ls='-.')
    # plt.legend(loc='upper right')
    # plt.ylabel('(dB)')
    # plt.xlabel('Distance (km)')
    # plt.ylim([35, 65])
    # plt.tight_layout()

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot((np.arange(range_domain[0], range_domain[1])-EFFECTIVE_LENGTH//2)*0.375/1000,
             rd_db4[::-1, slice_velocity], label='with compensation')
    ax.plot((np.arange(range_domain[0], range_domain[1])-EFFECTIVE_LENGTH//2)*0.375/1000,
             dd_db3[::-1, slice_velocity], label='without compensation')
    ax.grid(ls='-.')
    ax.legend(loc='upper right')
    ax.set_ylabel('(dB)')
    ax.set_xlabel('Distance (km)')
    ax.set_ylim([30, 75])

    if matched:
        axins = zoomed_inset_axes(ax, 5.5, loc=9)
        axins.plot((np.arange(range_domain[0], range_domain[1])-EFFECTIVE_LENGTH//2)*0.375/1000,
                 rd_db4[::-1, slice_velocity], label='with compensation')
        axins.plot((np.arange(range_domain[0], range_domain[1])-EFFECTIVE_LENGTH//2)*0.375/1000,
                 dd_db3[::-1, slice_velocity], label='without compensation')
        axins.set_xlim([4.115, 4.135])
        axins.set_ylim([69.5, 72.75])
        axins.grid(ls=':')
        plt.xticks(visible=False)
        # axins.yaxis.tick_right()
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axins = zoomed_inset_axes(ax, 2.5, loc='upper left')
    axins.plot((np.arange(range_domain[0], range_domain[1])-EFFECTIVE_LENGTH//2)*0.375/1000,
             rd_db4[::-1, slice_velocity], label='with compensation')
    axins.plot((np.arange(range_domain[0], range_domain[1])-EFFECTIVE_LENGTH//2)*0.375/1000,
             dd_db3[::-1, slice_velocity], label='without compensation')
    axins.set_xlim([3.56, 3.68])
    axins.set_ylim([43.5, 51])
    axins.grid(ls=':')
    plt.xticks(visible=False)
    axins.yaxis.tick_right()
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.tight_layout()

    if save_fig:
        plt.savefig('Doppler_slice_at_velocity_{:.3}mps_match_{}.png'.format(v[slice_velocity], str(matched)), dpi=300)


# %% 3db position around



# %% save results

save_data = False

if save_data:
    savemat('data_match_{}.mat'.format(str(matched)),
            mdict={
                "doppler_data3": doppler_data3,
                "range_data4": range_data4
            })

load_data = False

if load_data:
    mdata = loadmat('data_match_{}.mat'.format(str(matched)))
    doppler_data3 = mdata['doppler_data3']
    range_data4 = mdata['range_data4']
# -*- coding: utf-8 -*-
'''
Creat on 14/10/2019

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from tqdm import tqdm
from numpy import exp, pi, log10
from scipy.signal import deconvolve, convolve

from PyMat.DataProcess import (slow_time_fft,
                               fast_time_correlation,
                               doppler_compensation,
                               load_matched_code,
                               range_doppler_show)
from PyMat.SysParas import *
from PyMat.DataReform import data_reform

# original = [1j, 1, 0, 0, 1, 1, 0, 0]
# impulse_response = [2+1j, 1]
# recorded = convolve(impulse_response, original)
# recorded_same = convolve(impulse_response, original, 'same')
# print(recorded)
#
# recovered, remainder = deconvolve(recorded, impulse_response)
# print(recovered.real.astype('int'))
# print(recovered.imag.astype('int'))

def range_deconv(data, matched_code):
    print("\nRange deconvolution starts ...")
    range_recover = np.zeros((EFFECTIVE_LENGTH, data.shape[1]), dtype='complex')
    for i in tqdm(range(data.shape[1])):
        range_recover[:, i] = deconvolve(data[:, i], matched_code)

    print("Range deconvolution finished!")
    return range_recover


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
    path =  '/Volumes/Personal/Backup/PMCWPARSAXData/cars/'
    file ='HH_20190919134453.bin'
    target = 'cars'

    # path = '/Volumes/Personal/Backup/PMCWPARSAXData/chimney/'
    # file ='HH_20190919075858.bin'
    # target = 'chimney'
    #
    # path =  '/Volumes/Personal/Backup/PMCWPARSAXData/A13/'
    # file = 'VV_20191009081112.bin'
    # target = 'vehicles'

    N_Block = 61
    recei, trans = data_reform(path + file,
                               n_block_to_process=N_Block,
                               verbose=False,
                               filter=True,
                               win_func='rect')

    fast_time, slow_time = recei.shape
    # global EFFECTIVE_LENGTH
    # EFFECTIVE_LENGTH = 262144//int(1e-3/PERIOD_DURATION)
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
                                        conv_mode='full')
    doppler_data1 = slow_time_fft(range_data1,
                                  win_func=win_func,
                                  shifted=True,
                                  fft_zoom=fft_zoom)

    effective_start = EFFECTIVE_LENGTH//2 - 1
    range_domain = [effective_start+1500, effective_start+5500]

    range_doppler_show(doppler_data1,
                       start_plot=range_domain[0],
                       end_plot=range_domain[1],
                       normalize=False,
                       clim=60,
                       title="{}_range_doppler".format(target),
                       save_fig=save_fig,
                       downsampling_rate=downsampling_rate
                       )

    doppler_code = range_deconv(doppler_data1, matched_code)

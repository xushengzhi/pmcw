# -*- coding: utf-8 -*-
'''
Creat on 16/09/2019

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.io import loadmat
from scipy.constants import speed_of_light as c
from numpy import pi, exp, log10
from numpy.fft import fft, fftshift

from PyMat.SysParas import *

save_code = False

# %% Load Code
code = loadmat('code.mat')['codes']
code1 = code[0, :]
code2 = code[1, :]
code_length = code.shape[1]
print('The code length is {}'.format(code_length))
del code
# code1 = np.ones_like(code1)
# code2 = np.ones_like(code2)

# %% Radar parameters
sampling_frequency = 1.2e9
sampling_interval = 1/sampling_frequency
Tc = 2/BANDWIDTH
code_repeat = 1
PRI = 1e-3


# %% Generate code
duration_grid = int(Tc/sampling_interval)                  # 48
T = Tc * code_length * code_repeat                         # 3.2768 ms
print('PRI: ', T)
vm = c/4/CARRIER_FREQUENCY/PRI
Rm = c*T/2
print('vm: {}\nRm: {}'.format(vm, Rm))
dR = c/2/BANDWIDTH

code_length_for_periods = code_length * code_repeat
repeat_time_in_each_period = code_repeat

code1_rep = np.tile(code1, repeat_time_in_each_period)
code2_rep = np.tile(code2, repeat_time_in_each_period)

code1_tc = np.repeat(code1_rep[0:code_length_for_periods], duration_grid)
code2_tc = np.repeat(code2_rep[0:code_length_for_periods], duration_grid)
period_length = code1_tc.size

center_wave = exp(2j*pi*INTERMEDIATE_FREQUENCY*np.arange(period_length)*sampling_interval)


Tz = PRI - T                           # Tz for zeros
Nz = int(np.ceil(Tz / Tc * duration_grid))

transmit_period1 = np.zeros((code1_tc.size + Nz, ), dtype=complex)
transmit_period1[0:code1_tc.size] = center_wave * code1_tc

transmit_period2 = np.zeros((code2_tc.size + Nz, ), dtype=complex)
transmit_period2[0:code2_tc.size] = center_wave * code2_tc
# %%
plt.figure()
plt.plot(code1_tc)
plt.figure()
plt.plot(transmit_period1.real)

plt.figure()
plt.plot(np.linspace(-600, 600, transmit_period1.size, endpoint=False),
         (20*log10(abs(fftshift(fft(transmit_period1.real))) + 1e-50)))
plt.xlim([0, 300])
plt.ylim([0, 100])
plt.grid(ls=':')
plt.xlabel('Frequency MHz')

print(transmit_period1.size)

# %% write text file
output = 'cw.txt'
if save_code:
    import os
    if os.path.exists(output):
        ans = input('The file {} already exists.\nDo you want to replace it? (y/n)'.format(output))

        if ans.lower() == 'y':
            os.remove(output)

        else:
            print('The code is not saved!')
    else:
        generator = open(output, 'w+')
        for i in tqdm(range(transmit_period1.size)):
            generator.write(str(transmit_period1.real[i]))
            generator.write(', 0, 0\n')
        generator.close()


# %% test filter
st = 32
trans_p = np.tile(transmit_period1.real[::3], [st, 1]).flatten()
from PyMat.DataReform import filtering
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
trans_d, _ = filtering(trans_p, fs=400e6, win_func='rect')

plt.figure()
plt.plot(np.linspace(-200, 200, trans_d.size, endpoint=False),
         (20*log10(abs(fftshift(fft(trans_d))) + 1e-50)))
plt.grid(ls=':')
plt.ylim([-10, 110])
plt.xlim([-50, 50])
plt.xlabel('Frequency MHz')

trans_dn = trans_d * exp(2j*pi*0.00002*np.arange(trans_d.size)) + 0.1*np.random.randn(trans_d.size)
trans_dr = trans_dn.reshape([st, 400_000]).T

plt.figure()
plt.imshow(20*log10(abs(fftshift(fft(trans_dr[0:1000, :], n=128, axis=-1), axes=-1))),
           aspect='auto',
           cmap='jet',
           extent=[-0.5, 0.5, 0, 999]
           )





















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


save_code = True

# %% Load Code
code = loadmat('code.mat')['codes']
code1 = code[0, :]
code2 = code[1, :]
code_length = code.shape[1]
print('The code length is {}'.format(code_length))
del code

# %% Radar parameters
center_frequency = 125e6
carrier_frequency = 3.315e9
sampling_frequency = 1.2e9
sampling_interval = 1/sampling_frequency
bandwidth = 50e6
Tc = 2/bandwidth
code_repeat = 1


# %% Generate code
duration_grid = int(Tc/sampling_interval)                  # 48
T = Tc * code_length * code_repeat
print('PRI: ', T)
vm = c/4/carrier_frequency/T
Rm = c*T/2
print('vm: {}\nRm: {}'.format(vm, Rm))
dR = c/2/bandwidth

code_length_for_periods = code_length * code_repeat
repeat_time_in_each_period = code_repeat

code1_rep = np.tile(code1, repeat_time_in_each_period)
code2_rep = np.tile(code2, repeat_time_in_each_period)

code1_tc = np.repeat(code1_rep[0:code_length_for_periods], duration_grid)
period_length = code1_tc.size

center_wave = exp(2j*pi*center_frequency*np.arange(period_length)*sampling_interval)


Tz = 1e-3 - T                           # Tz for zeros
Nz = int(np.ceil(Tz / Tc * duration_grid))

transmit_period = np.zeros((code1_tc.size + Nz, ), dtype=complex)
transmit_period[0:code1_tc.size] = center_wave * code1_tc
# %%
plt.figure()
plt.plot(code1_tc[0:200])
plt.figure()
plt.plot(transmit_period.real[0:200])

plt.figure()
plt.plot(np.linspace(-600, 600, transmit_period.size), (20*log10(abs(fftshift(fft(transmit_period.real))) + 1e-50)))
plt.xlim([0, 300])
plt.ylim([0, 100])
plt.grid(ls=':')
plt.xlabel('Frequency MHz')

# %% write text file
output = 'pmcw_waveform.txt'
if save_code:
    import os
    if os.path.exists(output):
        ans = input('The file {} already exists.\nDo you want to replace it? (y/n)'.format(output))

        if ans.lower() == 'y':
            os.remove(output)

            generator = open(output, 'w+')
            for i in tqdm(range(transmit_period.size)):
                generator.write(str(transmit_period.real[i]))
                generator.write(', 0, 0\n')
            generator.close()

        else:
            print('The code is not saved!')























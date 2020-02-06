# -*- coding: utf-8 -*-
'''
Creat on 16/09/2019

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm import tqdm
from scipy.io import loadmat
from scipy.constants import speed_of_light as c
from numpy import pi, exp, log10
from numpy.fft import fft, fftshift

from SysParas import BANDWIDTH, CARRIER_FREQUENCY, INTERMEDIATE_FREQUENCY, SAMPLING_FREQUENCY

mpl.rcParams['agg.path.chunksize'] = 10000


save_code = False
save_fig = False
# %% Load Code
# load mutual orthogonal codes series code1 and code2
code = loadmat('code4096.mat')['codes']
code1 = code[0, :]
code2 = code[1, :]
code_length = code.shape[1]
print('The code length is {}'.format(code_length))
del code

# %% Radar parameters
# radar parameters setting, the sampling frequency of the waveform generator
sampling_frequency = SAMPLING_FREQUENCY                  # Waveform Generator Sampling Frequency
sampling_interval = 1/sampling_frequency
Tc = 2/BANDWIDTH
PRI = 2e-4

# Reset parameters
duration_grid1 = int(np.floor(sampling_frequency*PRI / code_length))     # maximum number of samples for one bchip
duration_grid2 = int(np.ceil(Tc/sampling_interval))                      # according to code length and bandwidth
duration_grid = np.min([duration_grid1, duration_grid2])
Tc = sampling_interval * duration_grid
real_bandwidth = 2/Tc
print('Real bandwidth: {:>17.1f}'.format(real_bandwidth))
print('Smaples for each Bcode: {:>8}'.format(duration_grid))
T = Tc * code_length                                                     # 0.65 ms
print('bcode duration time: {:>11.6f}'.format(T))

vm = c/4/CARRIER_FREQUENCY/PRI
Rm = c*T/2
dR = c/2/BANDWIDTH
print('vm: {:>28.3f}\nRm: {:>28.3f}\ndR: {:>28.3f}'.format(vm, Rm, dR))

# %% Generate code
samples_for_code = code_length * duration_grid
samples_per_pri = int(sampling_frequency * PRI)
samples_zeros = samples_per_pri - samples_for_code
time_zeros = samples_zeros * sampling_interval

code1_rep = np.repeat(code1, duration_grid)
code2_rep = np.repeat(code2, duration_grid)

trans_code1 = np.zeros(samples_per_pri, )
trans_code2 = np.zeros(samples_per_pri, )

trans_code1[0: samples_for_code] = code1_rep
trans_code2[0: samples_for_code] = code2_rep

# %% mix with the carrier wave
center_wave = exp(2j*pi*INTERMEDIATE_FREQUENCY*np.arange(samples_per_pri)*sampling_interval)
transmit_period1 = center_wave * trans_code1
transmit_period2 = center_wave * trans_code2


# %% plot
save_fig = False
plt.figure(figsize=[15, 3])
plt.step(np.arange(400), trans_code1[0:400])
plt.xlabel('fast time index')
plt.ylabel('amplitude')
# plt.title('binary codes')
plt.tight_layout()
if save_fig:
    plt.savefig('generated_bcodes.png', dpi=300)

plt.figure(figsize=[10, 3])
plt.plot(transmit_period1.real[0:200])
plt.xlabel('fast time index')
plt.ylabel('amplitude')
plt.title('phase-coded waves')
plt.tight_layout()
if save_fig:
    plt.savefig('transmitted_mcodes.png', dpi=300)


plt.figure(figsize=[8, 4])
plt.plot(np.linspace(-sampling_frequency/2e6, sampling_frequency/2e6, 
                     transmit_period1.size, endpoint=False),
         (20*log10(abs(fftshift(fft(transmit_period1.real))) + 1e-50)))
plt.xlim([0, sampling_frequency/2e6])
plt.ylim([-20, 80])
plt.grid(ls=':')
plt.xlabel('Frequency MHz')
plt.ylabel('(dB)')
plt.tight_layout()
# plt.title('spectrum')
if save_fig:
    plt.savefig('spectrum_mcodes.png', dpi=300)

print(transmit_period1.size)

# %% write text file
output = 'pmcw_4096_code.txt'
def _save_code(output, file):
    generator = open(output, 'w+')
    for i in tqdm(range(file.size)):
        generator.write(str(file.real[i]))
        generator.write(', 0, 0\n')
    generator.close()

if save_code:
    import os
    if os.path.exists(output):
        ans = input('The file {} already exists.\nDo you want to replace it? (y/n)'.format(output))

        if ans.lower() == 'y':
            os.remove(output)
            _save_code(output, transmit_period1)

        else:
            print('The code is not saved!')
    else:
        _save_code(output, transmit_period1)



# %% test filter

# st = 32
# trans_p = np.tile(transmit_period1.real[::3], [st, 1]).flatten()
# from DataReform import filtering
# trans_d, _ = filtering(trans_p, fs=400e6, win_func='rect')
#
# plt.figure()
# plt.plot(np.linspace(-200, 200, trans_d.size, endpoint=False),
#          (20*log10(abs(fftshift(fft(trans_d))) + 1e-50)))
# plt.grid(ls=':')
# plt.ylim([-10, 110])
# plt.xlim([-50, 50])
# plt.xlabel('Frequency MHz')
#
# trans_dn = trans_d * exp(2j*pi*0.00002*np.arange(trans_d.size)) + 0.1*np.random.randn(trans_d.size)
# trans_dr = trans_dn.reshape([st, 400_000]).T
#
# plt.figure()
# plt.imshow(20*log10(abs(fftshift(fft(trans_dr[0:1000, :], n=128, axis=-1), axes=-1))),
#            aspect='auto',
#            cmap='jet',
#            extent=[-0.5, 0.5, 0, 999]
#            )


plt.show()

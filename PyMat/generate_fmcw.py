# -*- codeing: utf-8 -*-
'''
Create on 2020/2/6

@author: shengzhixu
@email:sz.xu@hotmial.com
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import exp, pi
from numpy.fft import fft, fftshift
from tqdm import tqdm

from utils import dB
from SysParas import SAMPLING_FREQUENCY, BANDWIDTH

PRI = 1e-4
slope = BANDWIDTH/PRI
# Tc = 2/BANDWIDTH
sampling_interval = 1/SAMPLING_FREQUENCY
f0 = 125e6
fend = 175e6

L = int(SAMPLING_FREQUENCY * PRI)
sweep_length = int(SAMPLING_FREQUENCY * PRI)
time_series = np.arange(L)*sampling_interval
# fmcw = np.zeros_like(time_series).astype(complex)

fmcw = exp(2j*pi*( f0*time_series + 0.5*slope*time_series**2 ))

# %% spectrum
plt.figure(figsize=[8, 4])
plt.plot(np.linspace(-SAMPLING_FREQUENCY*PRI/2, SAMPLING_FREQUENCY*PRI/2, L, endpoint=False),
         dB(fftshift(fft(fmcw))))
plt.xlim([0, SAMPLING_FREQUENCY*PRI/2])
plt.ylim([-20, 80])
plt.grid(ls=':')
plt.xlabel('Frequency MHz')
plt.ylabel('(dB)')
plt.tight_layout()


# %% save code
save_code = 0
output = 'fmcw4096.txt'


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
            _save_code(output, fmcw.real)

        else:
            print('The code is not saved!')
    else:
        _save_code(output, fmcw.real)
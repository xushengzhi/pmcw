# -*- codeing: utf-8 -*-
'''
Create on 2020/2/5

@author: shengzhixu
@email:sz.xu@hotmial.com
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from numpy import exp, pi, sin, sqrt, cos
from scipy.signal import correlate, convolve

from utils import dB, random_code, conv_circ
# matplotlib.rcParams['text.usetex'] = True


save_fig = 0
# %%
def peak_energy_loss(code, frequency=None):
    if frequency is None:
        frequency = np.linspace(0, 0.5, 11, endpoint=True)
    code_length = code.size
    E = dB(code_length)
    peak_energy_loss_array = np.zeros_like(frequency)
    sidelobe_energy_incre_array = np.zeros_like(frequency)
    time_series = np.arange(code_length) / code_length

    corr = convolve(code, code[::-1].conj(), mode='same')
    max_sidelobe = np.max(dB(corr[code_length//2 + 10:code_length-code_length//4]))

    for i, f in enumerate(frequency):
        # alpha = np.random.randn() + 1j * np.random.randn()
        # alpha = alpha / abs(alpha)
        alpha = 1
        doppler = alpha*exp(2j * pi * f * time_series)
        receive_code1 = code * doppler
        corr = convolve(receive_code1, code[::-1].conj(), mode='same')
        peak_energy_loss_array[i] = E - np.max(dB(corr))

        sidelobe_energy = np.max(dB(corr[code_length//2+10:code_length-code_length//4]))
        sidelobe_energy_incre_array[i] = sidelobe_energy - max_sidelobe

    return -peak_energy_loss_array, sidelobe_energy_incre_array



# %% Different codes
frequency = np.linspace(0, 0.5, 11, endpoint=True)
# frequency = [0.2372]
# ZCZ codes 4096
zcz = loadmat('data/code2048.mat')['codes'][0, :]
zcz_loss, zcz_inc = peak_energy_loss(zcz, frequency)

# apas code 1020
apas = loadmat('data/apas1020.mat')['codes'][0, :]
apas_loss, apas_inc = peak_energy_loss(apas, frequency)

# random codes 1024
random_1024 = random_code(1024)
random_1024_loss, random_1024_inc = peak_energy_loss(random_1024, frequency)

# random codes 2048
random_2048 = random_code(2048)
random_2048_loss, random_2048_inc = peak_energy_loss(random_2048, frequency)


# %% Plot
plt.figure(figsize=[8, 6])
plt.subplot(211)
plt.plot(frequency, zcz_loss, lw=2, marker=7, ms=13, label='ZCZ2048')
plt.plot(frequency, apas_loss, lw=2, marker=6, ms=13, label='APAS1020')
plt.plot(frequency, random_1024_loss, lw=2, marker='x', ms=19, label='Rand1024')
plt.plot(frequency, random_2048_loss, lw=2, marker='+', ms=19, label='Rand2048')
# therotical db_value
theoretical = np.zeros_like(zcz_loss)
theoretical[0] = 0
theoretical[1::] = dB(sqrt(2 - 2*cos(2*pi*frequency[1::]))/2/pi/frequency[1::])
plt.plot(frequency, theoretical, lw=2, marker='o', ms=9, label='Theoretical value')
plt.legend(loc='lower left')
plt.grid(ls=':')
# plt.xlabel(r'$\sigma \displaystyle\frac{v}{v_{\textit{max}}}$')
# plt.xlabel(r'$\nu$')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('Peak energy loss (dB)')
# plt.tight_layout()

# if save_fig:
#     plt.savefig('Peak_energy_loss.png', dpi=300)
plt.subplot(212)
# plt.figure(figsize=[8, 4])
plt.plot(frequency, zcz_inc, lw=2, marker=7, ms=13, label='ZCZ2048')
plt.plot(frequency, apas_inc, lw=2, marker=6, ms=13, label='APAS1020')
plt.plot(frequency, random_1024_inc, lw=2, marker='x', ms=13, label='Rand1024')
plt.plot(frequency, random_2048_inc, lw=2, marker='+', ms=13, label='Rand2048')
plt.legend(loc='upper left')
plt.grid(ls=':')
# plt.xlabel(r'$\sigma \displaystyle\frac{v}{v_{\textit{max}}}$')
plt.xlabel(r'$\nu$')
plt.ylabel('Sidelobe energy increment (dB)')
plt.tight_layout()
#
save_fig = 0
if save_fig:
    plt.savefig('Energy_variation.png', dpi=300)



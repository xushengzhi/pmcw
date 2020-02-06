# -*- coding: utf-8 -*-
'''
Creat on 29/01/2020

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

from AmbiguityFunc import ambiguity_function, ambiguity_function2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import dB, conv_circ


# %%
save_fig = False


# %%
def welch_bound(codes):
    s, l = codes.shape
    return np.sqrt((s - 1) * l**2 / (s*l -1))

codes = loadmat('code8192.mat')['codes']
code1 = codes[0, :]

# # np.random.shuffle(code1)
#
# code = np.random.rand(64)
# code1 = np.zeros_like(code)
# code1[code > 0.5] = -1
# code1[code <= 0.5] = 1
code1 = np.kron(code1, np.ones((16, )))
#
# fs = 1
# f_vec = np.linspace(-0.5, 0.5, 256, endpoint=False) * fs
# code_len = code1.size
#
# # af = ambiguity_function2(code1, f_vec, fs, mode='zero-padding')
# af = ambiguity_function2(code1, f_vec, fs, mode='shift')
#
# # %%
# plt.figure()
# plt.imshow(dB(af),
#            aspect='auto',
#            extent=[-code_len+1, code_len, f_vec[0], f_vec[-1]],
#            cmap='jet')
# plt.clim([-60, 0])
# plt.colorbar()

acf = (np.convolve(code1, code1[::-1], mode='same'))
acf_circ = np.roll(conv_circ(code1, code1[::-1]), 1)
welch = welch_bound(codes)

plt.figure()
plt.plot(np.arange(-65536, 65536), acf, lw=3, label='ACF', alpha=0.6)
plt.plot(np.arange(-65536, 65536), acf_circ, lw=3, label='CACF', alpha=0.6)
plt.hlines(welch, -65536, 65536, colors='r', lw=3, label='Welch Bound')
plt.legend(loc='upper right')
plt.xlabel('Index lag')
plt.ylabel('Autocorrelation')
if save_fig:
    plt.savefig('aotocorrelation_scale.png', dpi=300)


acf_dB = dB(np.convolve(code1, code1[::-1], mode='same'))
acf_circ_dB = dB(np.roll(conv_circ(code1, code1[::-1]), 1))
welch_dB = dB(welch_bound(codes))
max_dB = max(np.max(acf_dB), np.max(acf_circ_dB))
max_acf_side_dB = (np.max(acf_dB[0:32770]))
max_acf_circ_side_dB = (np.max(acf_circ_dB[0:32770]))

plt.figure(figsize=[8, 4])
p1 = plt.plot(np.arange(-65536, 65536), acf_dB-max_dB, lw=3, label='ACF', alpha=0.6)
plt.hlines(max_acf_side_dB-max_dB, -65536, -32768, lw=3, color=p1[0].get_color(), linestyles='--', alpha=1)
p2 = plt.plot(np.arange(-65536, 65536), acf_circ_dB-max_dB, lw=3, label='CACF', alpha=0.6)
plt.hlines(max_acf_circ_side_dB-max_dB, -65536, -32768, lw=3, color=p2[0].get_color(), linestyles='--', alpha=1)
plt.hlines(welch_dB-max_dB, -65536, 65536, colors='black', lw=3, label='Welch Bound', alpha=1)
plt.ylim([-80, 5])
plt.legend(loc='upper right')
plt.xlabel('Index lag')
plt.ylabel('Autocorrelation (dB)')
plt.xlim([-65536, 65536])
plt.grid(axis='y', linestyle=':')
plt.tight_layout()
if save_fig:
    plt.savefig('aotocorrelation_dB.png', dpi=300)

# -*- coding: utf-8 -*-
'''
Creat on 14/10/2019

Authors: shengzhixu

Email: 

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.signal import convolve, deconvolve
from numpy.fft import fft, fftshift, ifft


a = np.random.randn(5) + 1j *np.random.randn(5)
b = np.random.randn(5) + 1j *np.random.randn(5)

c = convolve(a, b[::-1].conj(), mode='same')
cc = convolve(a[::-1], b.conj(), mode='same')

d = convolve(a.conj(), b[::-1], mode='same')
e = convolve(b, a[::-1].conj(), mode='same')
f = convolve(b.conj(), a[::-1], mode='same')
g = convolve(a[::-1], b.conj(), mode='same')

print('c', c)
print('cc', cc)
print('d', d)
print('e', e)
print('f', f)
print('g', g)

# c = np.correlate(a, b, mode='valid')
# d = np.correlate(b, a, mode='valid')
# print('c', c)
# print('d', d)



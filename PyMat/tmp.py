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


x = np.array([1, 2, 3, 4, 2, 1, 10])
h = np.array([1, 1, 1, 2, 3, 2, 1j])

y = convolve(x, h)
print('y', y)

X = fft(x, 16)
H = fft(h, 16)
Y = X*H
yy = ifft(Y)
print('yy', yy.real)

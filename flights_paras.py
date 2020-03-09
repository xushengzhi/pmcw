#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:05:57 2020

@author: shengzhixu
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.constants import speed_of_light as c

'''
PARSAX observing flight parameters
'''

v = 1000/3.6
f0 = 3.315e9
B = 50e6

Tc = 2/B
T = c/4/v/f0

print(int(T/Tc))

#T/Tc = 6782, so the code length could be 4096
code_length = 4096
Tp = code_length * Tc   # 163.84 mus
T = 250.00              #mus

fd = 2*v/c *f0 
# -*- codeing: utf-8 -*-
'''
Create on 2020/3/9

@author: shengzhixu
@email:sz.xu@hotmial.com
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from SysParas import SAMPLING_FREQUENCY


# old pack: 0x62200
time = 1.311e-3
samples = time * SAMPLING_FREQUENCY
print( )
print(hex(int(samples)))
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.constants import speed_of_light as c

file = '/Users/shengzhixu/Documents/ExperimentalData/pmcw/cars/HH_20190919134453.bin'

with open(file, 'rb') as fid:
    info = np.fromfile(fid, count=2, dtype=np.int32)
    
    print(info[0])

    block_to_skip = 1
    n_block_to_process = 5

    if block_to_skip >0:
        for i in range(block_to_skip):
            A = np.fromfile(fid, count=info[0], dtype=np.int16)
            print(A.shape)
            
    A = np.fromfile(fid, count=info[0], dtype=np.int16)
    A = np.delete(A, np.s_[0:4])
    print(A.shape)

    for i in range(n_block_to_process-1):
        B = np.fromfile(fid, count=info[0], dtype=np.int16)[0]
        np.delete(B, np.s_[0:4])
        A = np.concatenate(A, B)
        
#print(A.size)
        
        
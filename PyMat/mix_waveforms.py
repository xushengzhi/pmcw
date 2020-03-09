# -*- codeing: utf-8 -*-
'''
Create on 2020/2/6

@author: shengzhixu
@email:sz.xu@hotmial.com
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_matched_code

# %% load codes
code_length = 2048

file_pmcw = 'data/pmcw_{}_zeropad_False.txt'.format(str(code_length))
file_pmcw_mismatch = 'data/pmcw_{}_mismatch_zeropad_False.txt'.format(str(code_length))
# file_fmcw = 'data/fmcw{}.txt'.format(str(code_length))

pmcw = load_matched_code(file_pmcw, filtering=False, compensated=False, verbose=False, win_func='rect')
pmcw_mismatch = load_matched_code(file_pmcw_mismatch, filtering=False, compensated=False, verbose=False, win_func='rect')
# fmcw = load_matched_code(file_fmcw, filtering=False, compensated=False)

# %% mix codes
# mix pmcw codes
mix_code = pmcw + pmcw_mismatch

# # mix pmcw and fmcw
# mix_code = pmcw + fmcw

# %% save code
save_code = 0
output = 'data/mix_pmcw_{}_zeropad_False.txt'.format(str(code_length))

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
            _save_code(output, mix_code)

        else:
            print('The code is not saved!')
    else:
        _save_code(output, mix_code)

# %%
mix_wave = load_matched_code(output, filtering=True, compensated=True, win_func='rect')
plt.plot(mix_wave.real)
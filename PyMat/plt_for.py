import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.io import loadmat
from tqdm import tqdm
from scipy.constants import speed_of_light as C
from scipy.io import savemat, loadmat
from numpy import pi, exp, log10
from numpy.fft import fft, fftshift, ifftshift, ifft
from scipy.signal import convolve

from DataReform import data_reform
from SysParas import *
from utils import load_matched_code, next_pow, conv_fft, deconv_fft

# %% images show
def range_doppler_show(data,
                       start_plot=0,
                       end_plot=8000,
                       downsampling_rate=1,
                       clim=40,
                       dB = True,
                       normalize=True,
                       cmap='jet',
                       title="Range_Doppler",
                       pic_title=None,
                       save_fig=False,
                       FIGSIZE=None):
    if FIGSIZE is None:
        FIGSIZE = [6, 7]
    print("\nRange-Doppler imaging started ...")
    dr = C / 2 / SAMPLING_FREQUENCY * downsampling_rate
    x = np.arange(EFFECTIVE_LENGTH) * dr
    vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION
    v = np.linspace(-vm, vm, data.shape[1], endpoint=False)
    plt.figure(figsize=FIGSIZE)
    if dB:
        data_db = 20 * log10(abs(np.flipud(data[start_plot:end_plot, :]))+1e-60)
    else:
        data_db = np.flipud(data[start_plot:end_plot, :])

    max_db = np.max(data_db)
    print("Maximum amplitude: {:.4}dB".format(max_db))
    if normalize:
        data_db = data_db - max_db
        max_db = 0
    plt.imshow(data_db,
               aspect='auto',
               cmap=cmap,
               extent=[v[0], v[-1], x[start_plot]/1000, x[end_plot]/1000])
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Range (km)')
    if type(clim) is int:
        plt.clim([max_db-clim, max_db])
    else:
        plt.clim(clim)
    if pic_title is not None:
        plt.title(pic_title)
    cbar = plt.colorbar()
    cbar.set_label("(dB)", rotation=-90, labelpad=14)


    A = 126.3
    B = 109.7

    # A = 132.0
    # B = 110.0

    # A = 72.1
    # B = 105.0

    # A = 73.1
    # B = 105.1

    plt.annotate('{}'.format(A), xy=(29.8, 3.837), xytext=(35.19, 3.755),
                arrowprops=dict(facecolor='b', shrink=0.05, width=1.0, headwidth=3.5),
                 size=12
                ) # '${:.1f}$'.format(data_db[739, 332])

    plt.annotate('{}'.format(B), xy=(29.8, 4.335), xytext=(35.19, 4.250),
                arrowprops=dict(facecolor='b', shrink=0.05, width=1.0, headwidth=3.5),
                 size=12
                )


    if save_fig:
        plt.savefig("{}.png".format(title), dpi=300)
    print("Range-Doppler imaging Finished")

    return data_db



# %% Pre-Setting
save_fig = False
CMAP = 'jet'
win_func = 'hann'

matched = True
if matched:
    match_code = 'pmcw_8192.txt'
else:
    match_code = 'pmcw_8192_miscode.txt'

target = 'cars'
PERIOD_DURATION = 1e-3/2
vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION

matched_code = load_matched_code(code_file=match_code, verbose=False, win_func='rect', compensated=True)
fft_zoom = 2
downsampling_rate = 1

range_domain = [9500, 12000]
mdata = loadmat('data_match_{}.mat'.format(str(matched)))
doppler_data3 = mdata['doppler_data3']
range_data4 = mdata['range_data4']
del mdata


dd_db3 = 20 * log10(abs(np.flipud(doppler_data3[9500:12000, :]))+1e-60)
rd_db4 = 20 * log10(abs(np.flipud(range_data4[9500:12000, :]))+1e-60)

v = np.linspace(-vm, vm, rd_db4.shape[1], endpoint=False)
slice_velocity = 325  # 75, 55, 332
print("velocity resolution: {} m/s".format(v[2] - v[1]))
print("Plot sliced data at velocity: {}".format(v[slice_velocity]))

fig = plt.figure(figsize=[15, 5])
ax = fig.add_subplot(2, 1, 1)
plt.plot(np.arange(range_domain[0], range_domain[1]) * 0.375 / 1000,
         rd_db4[::-1, slice_velocity], label='with compensation')
plt.plot(np.arange(range_domain[0], range_domain[1]) * 0.375 / 1000,
         dd_db3[::-1, slice_velocity], label='without compensation')
plt.legend()
plt.ylabel('(dB)')
# plt.xlabel('Range (km)')
plt.ylim([55, 125])
plt.grid(ls=':')
plt.tight_layout()

del range_data4, doppler_data3


matched = False
if matched:
    match_code = 'pmcw_8192.txt'
else:
    match_code = 'pmcw_8192_miscode.txt'

target = 'cars'
PERIOD_DURATION = 1e-3/2
vm = C / 4 / CARRIER_FREQUENCY / PERIOD_DURATION

matched_code = load_matched_code(code_file=match_code, verbose=False, win_func='rect', compensated=True)
fft_zoom = 2
downsampling_rate = 1

range_domain = [9500, 12000]
mdata = loadmat('data_match_{}.mat'.format(str(matched)))
doppler_data3 = mdata['doppler_data3']
range_data4 = mdata['range_data4']
del mdata

dd_db3 = 20 * log10(abs(np.flipud(doppler_data3[9500:12000, :]))+1e-60)
rd_db4 = 20 * log10(abs(np.flipud(range_data4[9500:12000, :]))+1e-60)


ax = fig.add_subplot(2, 1, 2)
plt.plot(np.arange(range_domain[0], range_domain[1]) * 0.375 / 1000,
         rd_db4[::-1, slice_velocity], label='with compensation')
plt.plot(np.arange(range_domain[0], range_domain[1]) * 0.375 / 1000,
         dd_db3[::-1, slice_velocity], label='without compensation')
plt.legend()
plt.ylabel('(dB)')
plt.xlabel('Range (km)')
plt.ylim([55, 125])
plt.grid(ls=':')
plt.tight_layout()

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd
import numpy as np

import pickle


# define the parameters to be used
max_input_size = 1e6
max_output_size = 2e6
gen = 2
if gen == 2:
    opu_text = "Gen 2 (2kHz)"
elif gen == 3:
    opu_text = "Gen 3 (up to 8kHz)"

gpu_model = "NVIDIA Tesla V100 16GB"
cpu_model = "Intel Xeon E5-1650 v3 3.50GHz"
ram = 126

# load timings and compute frequencies for CPU and GPU
res = pickle.load(open('gpu_timings.pkl', 'rb'))
s = pd.Series(res)
x = 1 / s.unstack()
x_gpu = x.values

res = pickle.load(open('cpu_timings.pkl', 'rb'))
s = pd.Series(res)
x = 1 / s.unstack()
x_cpu = x.values

# experiments for CPU and GPU have been performed on the same ranges of input and output sizes
input_sizes = s.unstack().index.values
output_sizes = s.unstack().columns.values


def get_opu_speed(input_size, output_size, gen='2'):
    """
    Function returning the speed in kHz at which the OPU can be used for given input and output sizes.
    We assume input_size <= 1M and output_size <= 2M (i.e. the OPU can be used).
    """

    if gen == 2:
        return 2
    elif gen == 3:
        if input_size <= 145e3 and output_size <= 81e3:
            return 8
        elif input_size <= 290e3 and output_size <= 81e3:
            return 6.3
        elif input_size <= 583e3 and output_size <= 163e3:
            return 3.8
        elif input_size <= 1e6 and output_size <= 326e3:
            return 1.9
        elif input_size <= 1e6 and output_size <= 600e3:
            return 1
        else:
            return 0.34
    else:
        raise ValueError("gen must be 2 or 3")


# compute the matrix telling us which device is the fastest depending on input and output size
C = np.empty((len(input_sizes), len(output_sizes)), dtype=np.int)
for i, n in enumerate(input_sizes):
    for j, m in enumerate(output_sizes):
        cpu = x_cpu[i, j] / 1000
        gpu = x_gpu[i, j] / 1000

        if np.isnan(gpu) and np.isnan(cpu):  # if nor CPU nor GPU works, just check if OPU does
            if n <= max_input_size and m <= max_output_size:
                C[i, j] = 2  # OPU
            else:
                C[i, j] = 3  # nothing
            continue
        elif np.isnan(gpu):
            max_speed = cpu
        else:  # gpu is always nan when cpu is, so no need for another elif
            max_speed = max(cpu, gpu)

        if n <= max_input_size and m <= max_output_size and max_speed <= get_opu_speed(n, m, gen):
            C[i, j] = 2
        else:
            if x_cpu[i, j] < x_gpu[i, j]:
                C[i, j] = 1
            else:
                C[i, j] = 0

# in my experiments, we weren't interested in all the input and output sizes I computed
start_r, end_r = 2, 13  # for output size (rows in the graph)
start_c, end_c = 4, 13  # for input size (columns in the graph)

# take coordinates we're interested in and transpose so coordinates correspond to what we want in the graph
# that is: input size horizontally (columns) and output size vertically (rows)
D = C[start_c:end_c,start_r:end_r].T
a = np.maximum(x_cpu, x_gpu)[start_c:end_c, start_r:end_r]
n, m = a.shape

# we now correct the fact that np.maximum propagates NaNs
b = a.copy()
reduced_cpu = x_cpu[start_c:end_c, start_r:end_r]
for i in range(n):
    for j in range(m):
        if np.isnan(a[i,j]) and not np.isnan(reduced_cpu[i,j]):
            b[i,j] = reduced_cpu[i,j]
b = b.T

# the actual plotting begins here
fig, ax = plt.subplots(figsize=(18, 18))

# depending on what you plot, you can have 3 or 4 colors, uncomment what you need
#cmap = ListedColormap(['lightblue', 'orange', 'green', 'black'])
cmap = ListedColormap(['lightblue', 'orange', 'green'])
cax = ax.matshow(D, cmap=cmap, origin='lower')

# the magic numbers make the bar as tall as the plot
#cbar = fig.colorbar(cax, ticks=np.linspace(0,3,9)[1::2], fraction=0.046, pad=0.04)
#cbar.ax.set_yticklabels((['CPU', 'GPU', 'OPU', 'None']), fontsize=12)
cbar = fig.colorbar(cax, ticks=np.linspace(0,2,7)[1::2], fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels((['CPU', 'GPU', 'OPU']), fontsize=12)

ax.xaxis.tick_bottom()
ax.set_xticks(np.arange(n))
ax.set_xticklabels(['{:.1e}'.format(s) for s in input_sizes[start_c:end_c]], fontsize=14)
ax.set_yticks(np.arange(m))
ax.set_yticklabels(['{:.1e}'.format(s) for s in output_sizes[start_r:end_r]], fontsize=14)

ax.set_xlabel('Input size', fontsize=14)
ax.set_ylabel('Output size', fontsize=14)

title = "Frequency of matrix-vector multiplication in kHz, and speed-up provided by OPU"
ax.set_title('{}\nOPU: {}\nGPU: {}\nCPU: {} ({} RAM)'.format(title, opu_text, gpu_model, cpu_model, ram), fontsize=18)

for i in range(n):
    for j in range(m):
        # about the use of i and j: dude, trust me
        if D[j,i] == 2:
            # speedup provided by the OPU in the zone it is fastest
            c = 1000 * get_opu_speed(input_sizes[i+start_c], output_sizes[j+start_r], gen) / b[j,i]
        else:
            c = b[j,i] / 1000
        ax.text(i, j, '{:.4g}'.format(c), va='center', ha='center', fontsize=15)

plt.tight_layout()
fig.savefig('test.png')


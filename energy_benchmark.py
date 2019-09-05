import torch
import numpy as np
from numpy import empty
import scipy.stats
from gpu_energy.power_info import EnergyMonitor
import synthetic_opu
from synthetic_opu import OPUModulePyTorch
import time
dtype = torch.FloatTensor

period = 5 # How often should we read (instant) power usage in seconds.
e = EnergyMonitor(period=period)

repetitions = 30 # repete the hole experiment and take the mean


n = 320 # Number of points to be projected
d_list = [100,32000] # Their dimension
p_list = [100,32000] # Their targeted dimension

r = .5 # proportion of ones in the data



t_gpu = empty((len(d_list),len(p_list)))
E_gpu = empty((len(d_list),len(p_list)))
P_gpu = empty((len(d_list),len(p_list)))

t_cpu = empty((len(d_list),len(p_list)))


for i in range(len(d_list)):
    d = d_list[0]
    for j in range(len(p_list)):
        x = torch.distributions.Binomial(1,r).sample((n,d)).type(dtype)
        p = p_list[j]
        gpu_obj = OPUModulePyTorch( d, p, initial_log_scale='auto', tunable_kernel=False, dtype=dtype)
        e1 = e.energy()
        for repe in range(repetitions):
            _ = gpu_obj(x+0.0*(repe+100*i+10000*j))
        e2 = e.energy()
        e_gpu = (e2-e1).select_gpu("0")
        t_gpu[i,j] = e_gpu.duration()
        E_gpu[i,j] = e_gpu.consumption()
        
print(t_gpu)
print(E_gpu)



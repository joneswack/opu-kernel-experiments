import torch
import numpy as np
from numpy import empty, Inf
import scipy.stats
from gpu_energy.power_info import EnergyMonitor
import synthetic_opu
from synthetic_opu import OPUModulePyTorch
import time
dtype = torch.FloatTensor

period = 2 # How often should we read (instant) power usage in seconds.
e = EnergyMonitor(period=period)

repetitions = 3 # repeat the hole experiment and take the mean


results_dir = "gpu_energy/results/"
DEVICE = 1 # we handle divices at high level cause we read nvidia-smi command to get power info
torch.cuda.set_device(DEVICE)
gpu_device = torch.cuda.current_device()
ngpus_torch = torch.cuda.device_count()

print("PyTorch detects "+str(ngpus_torch)+" GPUs.")
print("Experiment on GPU "+str(gpu_device)+".")

if e._ngpus != ngpus_torch:
    print("Pytorch doesn't see all GPUs. The GPUs number in EnergyMonitor corresponds to what nvidia-smi gives.")

n = 320 # Number of points to be projected
d_list = [100,320,1000,3200,10000,32000,100000] # Their dimension
p_list = [100,320,1000,3200,10000,32000,100000] # Their targeted dimension

r = .5 # proportion of ones in the data



t_gpu = empty((len(d_list),len(p_list)))
E_gpu = empty((len(d_list),len(p_list)))
P_gpu = empty((len(d_list),len(p_list)))

t_cpu = empty((len(d_list),len(p_list)))

cpu_limit = 5

for i in range(len(d_list)):
    d = d_list[i]
    print("d = "+str(d),end=".\n")
    print("Sampling x...", end = " ")
    x = torch.distributions.Binomial(1,r).sample((n,d)).type(dtype)
    print("Done.")
    for j in range(len(p_list)):
        print("Adding x")
        x = x+0.0*(100*i+10000*j)
        print("Done")
        p = p_list[j]
        print("Building OPU object...")
        obj = OPUModulePyTorch( d, p, initial_log_scale='auto', tunable_kernel=False, dtype=dtype).to("cpu")
        print("Done.", end = " ")
        x = x.to("cpu")
        if i+j<cpu_limit:
            e1 = e.energy()
            for repe in range(repetitions):
                _ = obj(x.detach())
            e2 = e.energy()
            t_cpu[i,j] = (e2-e1).duration()
        else:
            t_cpu[i,j] = Inf
        x = x.to(gpu_device)
        obj.to(gpu_device)
        e3 = e.energy()
        for repe in range(repetitions):
            _ = obj(x.detach())
        e4 = e.energy()
        e_gpu = (e4-e3).select_gpu(str(DEVICE)) # nvdia-smi number
        t_gpu[i,j] = e_gpu.duration()/repetitions    # take the mean
        E_gpu[i,j] = e_gpu.consumption()/repetitions
        P_gpu[i,j] = E_gpu[i,j]/t_gpu[i,j]
        if j< len(p_list)-1 :
            print("p = "+str(p_list[j]),end=", ")
        else:
            print("p = "+str(p_list[j]),end=".\n")


print("CPU time:")
print(t_cpu)
print("GPU time:")
print(t_gpu)
print("GPU energy (J):")
print(E_gpu)
print("GPU average power (W):")
print(P_gpu)




t,p,gpu = e.to_numpy()


import matplotlib
matplotlib.rcParams['timezone'] = 'Europe/Paris'
import matplotlib.pyplot as plt
 
for i in range(len(gpu)):
    plt.plot(t[i,:],p[i,:],label="GPU "+gpu[i])
plt.legend()
plt.savefig(results_dir+"consumption.png")

np.save(results_dir+"cpu_time",t_cpu) 
np.save(results_dir+"gpu_time",t_gpu) 
np.save(results_dir+"gpu_energy",E_gpu) 

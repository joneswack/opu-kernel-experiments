import torch
import numpy as np
from numpy import empty, Inf
import scipy.stats
import sys
sys.path.append(".")
from power_info import EnergyMonitor

sys.path.append("../Tests Jonas/")
from random_features import OPUModulePyTorch
import time
dtype = torch.FloatTensor

period = 2 # How often should we read (instant) power usage in seconds.
e = EnergyMonitor(period=period)

repetitions = 80 # repeat the whole experiment and take the mean


results_dir = "./results/"
DEVICE = "cpu"
#DEVICE = 1  
# we handle devices at high level 'cause we read nvidia-smi command to get power info



try:
    torch.cuda.set_device(DEVICE)
    gpu_device = torch.cuda.current_device()
    ngpus_torch = torch.cuda.device_count()
except (NameError, ValueError):
    gpu_device = "None"
    ngpus_torch = 0

print("PyTorch detects "+str(ngpus_torch)+" GPUs.")
print("Experiment on GPU "+str(gpu_device)+".")

if e._ngpus != ngpus_torch:
    print("Pytorch doesn't see all GPUs. The GPUs number in EnergyMonitor corresponds to what nvidia-smi gives.")

n = 1000 # Number of points to be projected
d_list = [100,320,1000,3200,10000,32000] # Their dimension
p_list = [100,320,1000,3200,10000,32000] # Their targeted dimension

r = .5 # proportion of ones in the data



t_gpu = empty((len(d_list),len(p_list)))
E_gpu = empty((len(d_list),len(p_list)))
t_cpu = empty((len(d_list),len(p_list)))
t_lgpu = empty((len(d_list),len(p_list)))
E_lgpu = empty((len(d_list),len(p_list)))
t_lcpu = empty((len(d_list),len(p_list)))
cpu_limit = 5

for i in range(len(d_list)):
    d = d_list[i]
    print("d = "+str(d),end=".\n")
    print("Sampling x...", end = " ")
    x = torch.distributions.Binomial(1,r).sample((repetitions,n,d)).type(dtype).to("cpu")
    print("Done.")
    for j in range(len(p_list)):
        p = p_list[j]
        print("p = "+str(p),end=".\n")
        print("Building OPU object...")
        e1_lcpu = e.energy()
        obj = OPUModulePyTorch( d, p, initial_log_scale='auto', tunable_kernel=False, dtype=dtype).to("cpu")
        e2_lcpu = e.energy()
        t_lcpu[i,j] = (e2_lcpu-e1_lcpu).duration()
        print("Done.")


        if i+j<cpu_limit:
            print("Start matmuls with cpu...")
            e1_cpu = e.energy()
            for repe in range(repetitions):
                _ = obj(x[repe,:,:].detach())
            e2_cpu = e.energy()
            print("Done.")
            t_cpu[i,j] = (e2_cpu-e1_cpu).duration()/repetitions
        else:
            t_cpu[i,j] = Inf

        if ngpus_torch>0:
            print("Transfert to GPU...")
            e1_lgpu = e.energy()
            obj.to(gpu_device)
            e2_lgpu = e.energy()
            print("Done.")
            print("Start matmuls...")
            e1_gpu = e.energy()
            for repe in range(repetitions):
                _ = obj(x[repe,:,:].to(gpu_device).detach())
            e2_gpu = e.energy()
            print("Done.")
            e_lgpu = (e2_lgpu-e1_lgpu).select_gpu(str(DEVICE)) # nvdia-smi number
            t_lgpu[i,j] = e_lgpu.duration() + t_lcpu[i,j]    
            E_lgpu[i,j] = e_lgpu.consumption()

            e_gpu  = (e2_gpu-e1_gpu).select_gpu(str(DEVICE)) 
            t_gpu[i,j] = e_gpu.duration()/repetitions    # take the mean
            E_gpu[i,j] = e_gpu.consumption()/repetitions



print("CPU loading time:")
print(t_lcpu)
print("GPU loading time:")
print(t_lgpu)
print("GPU loading energy (J):")
print(E_lgpu)


print("CPU time:")
print(t_cpu)
print("GPU time:")
print(t_gpu)
print("GPU energy (J):")
print(E_gpu)





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

np.save(results_dir+"cpu_loadingtime",t_lcpu) 
np.save(results_dir+"gpu_loadingtime",t_lgpu) 
np.save(results_dir+"gpu_loadingenergy",E_lgpu) 

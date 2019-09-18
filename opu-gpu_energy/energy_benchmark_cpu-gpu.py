import torch
from torch import nn
import numpy as np
from numpy import empty, Inf
import scipy.stats
import sys
sys.path.append(".")
from power_info import EnergyMonitor
from warnings import warn

#sys.path.append("../Tests Jonas/")
#from random_features import OPUModulePyTorch
import time

class RandomProjectionModule(nn.Module):
    def __init__(self, input_features, output_features, mean=0., std=1.,
             dtype=None,device=None):
        super(RandomProjectionModule, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.empty(output_features, input_features,dtype=dtype,device=device), requires_grad=False)
        torch.nn.init.normal_(self.weight, mean, std)
        
    def forward(self, input):
        return input.mm(self.weight.t())

# Projection module:
class OPUModulePyTorch(nn.Module):
    def __init__(self, input_features, output_features, initial_log_scale='auto',
                    tunable_kernel=False, dtype=None,device=None):
        super(OPUModulePyTorch, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        if initial_log_scale == 'auto':
            initial_log_scale = -0.5 * np.log(input_features) # 1. / np.sqrt(input_features)
        
        self.proj_real = RandomProjectionModule(input_features, output_features,
                     mean=0., std=np.sqrt(0.5), dtype=dtype,device=device)
        self.proj_im = RandomProjectionModule(input_features, output_features,
                     mean=0., std=np.sqrt(0.5), dtype=dtype,device=device)
        
        self.log_scale = nn.Parameter(
            # log makes sure that scale stays positive during optimization
            torch.ones(1,dtype=dtype) * initial_log_scale,
            requires_grad=tunable_kernel
        )
        
    def forward(self, input):
        out_real = self.proj_real(input)
        out_img = self.proj_im(input)
        
        output = (out_real**2 + out_img**2)
        
        # we scale with the original scale factor (leads to kernel variance)
        return torch.exp(self.log_scale) * output

### META SETTINGS
# -------------
RESULTS_DIR = "./results/"
DEVICE = 1 if torch.cuda.is_available() else "cpu" # Choose your GPU to monitor (nvidia-smi number)
# we handle devices at high level 'cause we read nvidia-smi command to get power info
PERIOD = 2 # How often should we read (instant) power usage in seconds.
SAVE = len(RESULTS_DIR)>0 # Save measurements on disk in a numpy array.
# -------------

e = EnergyMonitor(period=PERIOD)
if DEVICE != "cpu":
    ngpus_torch = torch.cuda.device_count()
    if e._ngpus == ngpus_torch:
        torch.cuda.set_device(DEVICE)
        # If pytorch sees all what `nvidia-smi` have, they both use the same GPU IDs (I experienced).
        print("Experiments on GPU "+str(DEVICE)+".")
    else:
        warn("Pytorch doesn't see all GPUs. The GPUs number in EnergyMonitor corresponds to what `nvidia-smi` gives. You may adapt the variables ´DEVICE´ (nvidia-smi number) and ´device´ (torch device) in this script.")
    device = torch.cuda.current_device()
    # ´DEVICE´ is used for EnergyMonitor that use nvidia-smi GPU number AND
    # ´device´ is used by PyTorch. The user may make sure that these two number ar the same and eventually

else:
    device = "cpu"
    ngpus_torch = 0

    



### EXPERIMENT SETTINGS
# -------------
repetitions = 1 # repeat the whole experiment and take the mean. Hurt GPU memory (ctrl F "x.to(device)")
n = 1000 # Number of points to be projected
d_list = [100,320,1000,3200,10000,32000] # Their dimension
p_list = [100,320,1000,3200,10000,32000] # Their targeted dimension
r = .5 # proportion of ones in the data
dtype = torch.float32 #torch.FloatTensor
# -------------

t_gpu = empty((len(d_list),len(p_list)))
E_gpu = empty((len(d_list),len(p_list)))
t_lgpu = empty((len(d_list),len(p_list)))
E_lgpu = empty((len(d_list),len(p_list)))
cpu_limit = 5 # When we give up trying with CPU, from 0 to max(len(d_list),len(p_list))

### Run experiment
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
        e1_lgpu = e.energy()
        torch.cuda.synchronize(device=device)
        obj = OPUModulePyTorch( d, p, initial_log_scale='auto', tunable_kernel=False, dtype=dtype, device=device)
        torch.cuda.synchronize(device=device)
        e2_lgpu = e.energy()
        print("Done.")
        e_lgpu = (e2_lgpu-e1_lgpu)
        t_lgpu[i,j] = e_lgpu.duration()
        if ngpus_torch>0:
            E_lgpu[i,j] = e_lgpu.select_gpu(str(DEVICE)).consumption()
        else: # no GPU, no consumption
            E_lgpu[i,j] = 0


        if ngpus_torch>0 or (device=="cpu" and i+j<cpu_limit):
            print("Transfert to GPU...")
            e1_logpu = e.energy()
            torch.cuda.synchronize(device=device)
            obj.to(device)
            torch.cuda.synchronize(device=device)
            e2_logpu = e.energy()
            e1_lxgpu = e.energy()
            x_gpu = x.to(device)
            e2_lxgpu = e.energy()
            print("Done.")

            print("Start matmuls...")
            e1_gpu = e.energy()
            torch.cuda.synchronize(device=device)
            for repe in range(repetitions):
                _ = obj(x_gpu[repe,:,:])
            torch.cuda.synchronize(device=device)
            e2_gpu = e.energy()
            print("Done.")

            # Flag intervals
            e_logpu = (e2_logpu - e1_logpu)
            e_lxgpu = (e2_lxgpu - e1_lxgpu)
            e_gpu   = ( e2_gpu  -  e1_gpu)
            # Durations
            t_lgpu[i,j]  += e_logpu.duration() + e_lxgpu.duration()/repetitions
            t_gpu[i,j]   =   e_gpu.duration()/repetitions    # take the mean
            print(t_gpu[i,j])
            # Energies
            if device=="cpu":
                E_gpu[i,j]  = 0
            else:
                E_lgpu[i,j] += e_logpu.select_gpu(str(DEVICE)).consumption()
                E_lgpu[i,j] += e_lxgpu.select_gpu(str(DEVICE)).consumption()/repetitions
                E_gpu[i,j]  =  e_gpu.select_gpu( str(DEVICE) ).consumption()/repetitions 

        else:
            t_gpu[i,j] = Inf
            E_gpu[i,j] = 0


print("CPU and GPU loading time:")
print(t_lgpu)
print("GPU loading energy (J):")
print(E_lgpu)

print("CPU and GPU computation time:")
print(t_gpu)
print("GPU computation energy (J):")
print(E_gpu)


t,p,gpu = e.to_numpy()


if device != "cpu": # display energy consumption
    import matplotlib
    matplotlib.rcParams['timezone'] = 'Europe/Paris'
    import matplotlib.pyplot as plt
    
    for i in range(len(gpu)):
        plt.plot(t[i,:],p[i,:],label="GPU "+gpu[i])
    plt.legend()
    if SAVE:
        plt.savefig(RESULTS_DIR+"consumption.png")
    else:
        plt.show()

if SAVE:
    np.save(RESULTS_DIR+"gpu_time",t_gpu) 
    np.save(RESULTS_DIR+"gpu_energy",E_gpu) 

    np.save(RESULTS_DIR+"gpu_loadingtime",t_lgpu) 
    np.save(RESULTS_DIR+"gpu_loadingenergy",E_lgpu) 

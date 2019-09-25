import torch
from power_info import EnergyMonitor
from numpy import array, nan, save, Inf
from time import time
### META SETTINGS
# -------------
RESULTS_DIR = "results_energy_curves/"
DEVICE = 2 if torch.cuda.is_available() else "cpu" # Choose your GPU to monitor (nvidia-smi number)
# we handle devices at high level 'cause we read nvidia-smi command to get power info
PERIOD = 1 # How often should we read (instant) power usage in seconds.
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

def sync():
    if device != "cpu":
        torch.cuda.synchronize(device=device)

### EXPERIMENT SETTINGS
# -------------
repetitions = 8 # repeat the whole experiment and take the mean. Does't hurt GPU memory (ctrl-F "x.to(device)")
x_variations = 2 # alternate variations during the repetitions. Hurt GPU memory (ctrl-F "x.to(device)")
dry_runs = 2
n = 3000 # Number of points to be projected
d_list = tuple(i*2000+12000 for i in range(23)) # Their dimension 
# Dimension of their projections is also d as what matters for
# GPU times is input_dim * output_dim.
r = .5 # proportion of ones in the data
dtype = torch.float32 # torch.float16, even faster
only_real = True # real weight instead of complex numbers.
cpu_dlimit = Inf #12000
# -------------


res_time = array(tuple(nan for i in range(len(d_list))))
res_ener = array(tuple(nan for i in range(len(d_list))))

### Run experiment

for i in range(len(d_list)):
    d = d_list[i]
    if d<=cpu_dlimit or device != "cpu":
        p = d
        print("Sampling x...", end = " ")
        x = torch.distributions.Binomial(1,r).sample((x_variations,n,d)).type(dtype)
        print("Done.")

        print("Transfert "+str(x_variations)+" repetitions of a "+str(n)+"x"+str(d)+" matrix to GPU...")
        e1_lxgpu = time()
        sync()
        x_gpu = x.to(device)
        sync()
        e2_lxgpu = time()
        lxgpu = (e2_lxgpu-e1_lxgpu)/repetitions
        print("Done: "+str(round(lxgpu,6))+" s in average for one repetition.")

        print("Building the weight matrix...")
        e1_lgpu = time()
        sync()
        real = torch.randn(d,p,dtype=dtype,device=device) # complex weights
        if not only_real:
            imaginary = torch.randn(d,p,dtype=dtype,device=device)
        sync()
        e2_lgpu = time()
        lgpu = (e2_lgpu-e1_lgpu)
        print("Done: "+str(round(lgpu,6))+" s.")

        if dry_runs>0:
            print("Start warmup (dry runs)...")
            e1_cgpu = time()
            sync()
            for repe in range(dry_runs):
                if only_real:
                    _ = (x_gpu[repe%x_variations,...].mm(real))**2
                else:
                    _ = (x_gpu[repe%x_variations,...].mm(real))**2 + (x_gpu[repe%x_variations,...].mm(imaginary))**2
            sync()
            e2_cgpu = time()
            cgpu = (e2_cgpu-e1_cgpu)/dry_runs
            print("Done: "+str(round(cgpu,6))+" s in average for one repetition.")

        print("Start measured matmuls...")
        e1_cgpu = e.energy()
        sync()
        for repe in range(repetitions):
            if only_real:
                _ = (x_gpu[repe%x_variations,...].mm(real))**2
            else:
                _ = (x_gpu[repe%x_variations,...].mm(real))**2 + (x_gpu[repe%x_variations,...].mm(imaginary))**2
        sync()
        e2_cgpu = e.energy()
        del x
        del x_gpu
        del real
        if not only_real:
            del imaginary
        torch.cuda.empty_cache()
        cgpu = (e2_cgpu-e1_cgpu).duration()/repetitions
        print("Done: "+str(round(cgpu,6))+" s in average for one repetition.")

        frequence = n/cgpu

        print("The frequence of "+str(p)+"x"+str(d)+" matrix-vector multiplication is "+str(round(frequence/1000,6))+" kHz.")

        opu_frequency = 2000 
        print("Speed-up of an OPU at "+str(round(opu_frequency/1000,6))+" kHz: "+str(round(opu_frequency/frequence,6))+".")

        res_time[i] = cgpu
        res_ener[i] =  (e2_cgpu-e1_cgpu).select_gpu(str(DEVICE)).consumption()/repetitions

resname = "cpu_" if device == "cpu" else "gpu_"

save(RESULTS_DIR+resname+"time",res_time)
save(RESULTS_DIR+resname+"ener",res_ener)
save(RESULTS_DIR+"dimensions",array(d_list)) 

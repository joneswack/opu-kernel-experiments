import torch
import time
from numpy import empty

if torch.cuda.is_available():
    iscuda = True
    torch.cuda.set_device(2)
    device = torch.cuda.current_device()
else:
    iscuda = False
    device = "cpu"

def sync():
    if iscuda:
        torch.cuda.synchronize(device=device)

### EXPERIMENT SETTINGS
# -------------
repetitions = 1 # repeat the whole experiment and take the mean. Hurt GPU memory (ctrl-F "x.to(device)")
n_list = tuple(10*(i+1) for i in range(90)) # Number of points to be projected
d_list = (10000,25000,32000) # Their dimension
p_list = (10000,25000,32000) # Dimension of their projections # [100,320,1000]#
r = .5 # proportion of ones in the data
dtype = torch.float32 # torch.float16, even faster
only_real = False # real weight instead of complex numbers.
# -------------

### Run experiment
f_gpu = empty((len(n_list),len(d_list),len(p_list)))

for i in range(len(n_list)):
    n = n_list[i]
    for j in range(len(d_list)):
        d = d_list[j]
        for k in range(len(p_list)):
            p = p_list[k]
            print("(n,d,p) = ("+str(n)+","+str(d)+","+str(p)+").")
            print("Sampling x...", end = " ")
            x = torch.distributions.Binomial(1,r).sample((repetitions,n,d)).type(dtype)
            print("Done.")
            print("Building the weight matrix...", end = " ")
            e1_lgpu = time.time()
            sync()
            real = torch.randn(d,p,dtype=dtype,device=device)
            if not only_real: # complex weights
                imaginary = torch.randn(d,p,dtype=dtype,device=device)
            sync()
            e2_lgpu = time.time()
            lgpu = e2_lgpu-e1_lgpu
            print("Done: "+str(round(lgpu,6))+" s.")
            print("Transfert "+str(repetitions)+" repetitions of a "+str(n)+"x"+str(d)+" matrix to GPU...")
            e1_lxgpu = time.time()
            sync()
            x_gpu = x.to(device)
            sync()
            e2_lxgpu = time.time()
            lxgpu = (e2_lxgpu-e1_lxgpu)/repetitions
            print("Done: "+str(round(lxgpu,6))+" s in average for one repetition.")
            print("Start matmuls...", end= " ")
            e1_cgpu = time.time()
            sync()
            for repe in range(repetitions):
                if only_real:
                    _ = (x_gpu[repe,...].mm(real))**2
                else:
                    _ = (x_gpu[repe,...].mm(real))**2 + (x_gpu[repe,...].mm(imaginary))**2
            sync()
            e2_cgpu = time.time()
            cgpu = (e2_cgpu-e1_cgpu)/repetitions
            print("Done: "+str(round(cgpu,6))+" s in average for one repetition.")
            frequence = n/cgpu
            print("The frequency of "+str(p)+"x"+str(d)+" matrix-vector multiplication is "+str(round(frequence/1000,6))+" kHz.")
            f_gpu[i,j,k] = frequence
            del x_gpu
            del real
            if not only_real:
                del imaginary
             # These dels really help my machine's management of memory.


import matplotlib
import matplotlib.pyplot as plt

opu_frequency = 2000 

for j in range(len(d_list)):
    for k in range(len(p_list)):
        plt.plot(n_list,f_gpu[:,j,k]/1000,label="(d,p) = ("+str(d_list[j])+","+str(p_list[k])+")")
plt.plot(n_list,tuple(opu_frequency/1000 for i in range(len(n_list))),label=str(round(opu_frequency/1000,6))+" kHz")
plt.legend()
plt.title("Frequency of p×d matrix-vector multiplications\nw.r.t. to the batchsize n (kHz)")
plt.xlabel("n")
plt.ylabel("f")
plt.yscale("log")
plt.savefig("./optimal_batchsize.png")

#
#print("Speed-up of an OPU at "+str(round(opu_frequency/1000,6))+" kHz: "+str(round(opu_frequency/frequence,6))+".")

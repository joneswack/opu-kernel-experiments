import torch
import time

if torch.cuda.is_available():
    torch.cuda.set_device(1)
    device = torch.cuda.current_device()
else:
    device = "cpu"

### EXPERIMENT SETTINGS
# -------------
repetitions = 1 # repeat the whole experiment and take the mean. Hurt GPU memory (ctrl-F "x.to(device)")
n = 3000 # Number of points to be projected
d = 55000 # Their dimension
p = 55000 # Dimension of their projections
r = .5 # proportion of ones in the data
dtype = torch.float32 # torch.float16, even faster
only_real = True # real weight instead of complex numbers.
# -------------

### Run experiment

print("Sampling x...", end = " ")
x = torch.distributions.Binomial(1,r).sample((repetitions,n,d)).type(dtype)
print("Done.")



print("Building the weight matrix...")
e1_lgpu = time.time()
real = torch.randn(d,p,dtype=dtype,device=device) # complex weights
if not only_real:
    imaginary = torch.randn(d,p,dtype=dtype,device=device)
e2_lgpu = time.time()
lgpu = e2_lgpu-e1_lgpu
print("Done: "+str(round(lgpu,6))+" s.")


print("Transfert "+str(repetitions)+" repetitions of a "+str(n)+"x"+str(d)+" matrix to GPU...")
e1_lxgpu = time.time()
x_gpu = x.to(device)
e2_lxgpu = time.time()
lxgpu = (e2_lxgpu-e1_lxgpu)/repetitions
print("Done: "+str(round(lxgpu,6))+" s in average for one repetition.")


print("Start matmuls...")
e1_cgpu = time.time()
for repe in range(repetitions):
    if only_real:
        _ = (x_gpu[repe,...].mm(real))**2
    else:
        _ = (x_gpu[repe,...].mm(real))**2 + (x_gpu[repe,...].mm(imaginary))**2
e2_cgpu = time.time()
cgpu = (e2_cgpu-e1_cgpu)/repetitions
print("Done: "+str(round(cgpu,6))+" s in average for one repetition.")

frequence = n/cgpu

print("The frequence of "+str(p)+"x"+str(d)+" matrix-vector multiplication is "+str(round(frequence/1000,6))+" kHz.")

opu_frequency = 2000 
print("Speed-up of an OPU at "+str(round(opu_frequency/1000,6))+" kHz: "+str(round(opu_frequency/frequence,6))+".")
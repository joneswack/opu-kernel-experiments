import time
import numpy as np
from numpy import empty

import torch
dtype = torch.FloatTensor

from lightonopu.opu import OPU


from lightonml.projections.sklearn import OPUMap

repetitions = 10 # repeat the whole experiment and take the mean

results_dir = "./results/"


n = 1000 # Number of points to be projected
d_list = [100,320,1000,3200,10000,32000,45000] # Their dimension
p_list = [100,320,1000,3200,10000,32000,45000] # Their targeted dimension

r = .5 # proportion of ones in the data

t_opu = empty((len(d_list),len(p_list)))
t_lopu = empty((len(d_list),len(p_list)))



for i in range(len(d_list)):
    d = d_list[i]
    print("d = "+str(d),end=".\n")
    print("Sampling x...", end = " ")
    x = torch.distributions.Binomial(1,r).sample((repetitions,n,d)).type(dtype).numpy().astype('uint8')
    print("Done.")
    for j in range(len(p_list)):
        p = p_list[j]
        print("p = "+str(p),end=".\n")
        print("Building OPU object...")
        e1_lopu = time.time()
        #opu = OPU(p)
        random_mapping = OPUMap(n_components=p, ndims=1)
        e2_lopu = time.time()
        t_lopu[i,j] = e2_lopu-e1_lopu
        print("Done.")

        print("Start matmuls...")
        e1_opu = time.time()
        for repe in range(repetitions):
            _ = random_mapping.transform(x[repe,:,:])
        e2_opu = time.time()
        print("Done.")
        t_opu[i,j] = (e2_opu-e1_opu)/repetitions


print("OPU loading time:")
print(t_lopu)
print("OPU time:")
print(t_opu)

np.save(results_dir+"opu_time",t_opu) 
np.save(results_dir+"opu_loadingtime",t_lopu) 

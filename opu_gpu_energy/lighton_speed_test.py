import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(1)
from time import time
import pickle

use_gpu = True
torch_instead_of_numpy = False  # if use_gpu is False (torch is usually slightly faster than numpy on CPU)
sizes = np.floor(np.logspace(0, 7, 15)).astype(np.int)
dry_runs = 10
averaged_runs = 5

res = {}

for input_size in sizes:
    print(input_size)
    try:
        if use_gpu:
            x = torch.cuda.FloatTensor(input_size).normal_()
        else:
            if torch_instead_of_numpy:
                x = torch.FloatTensor(input_size).normal_()
            else:
                x = np.random.randn(input_size).astype(np.float32)

        for output_size in sizes:
            print(output_size)
            if not use_gpu and output_size*input_size>100000:
                    res[(input_size, output_size)] = 1000000
            else:
                try:
                    if use_gpu:
                        M = torch.cuda.FloatTensor(output_size, input_size).normal_()
                    else:
                        if torch_instead_of_numpy:
                            M = torch.FloatTensor(output_size, input_size).normal_()
                        else:
                            M = np.random.randn(output_size, input_size).astype(np.float32)

                    for i in range(dry_runs):
                        if use_gpu or torch_instead_of_numpy:
                            y = torch.mv(M, x)
                        else:
                            y = np.dot(M, x)

                    if use_gpu:
                        torch.cuda.synchronize()

                    times = []
                    for i in range(averaged_runs):
                        t1 = time()
                        if use_gpu or torch_instead_of_numpy:
                            y = torch.mv(M, x)
                        else:
                            y = np.dot(M, x)

                        if use_gpu:
                            torch.cuda.synchronize()
                        t2 = time()
                        times.append(t2-t1)

                    res[(input_size, output_size)] = np.mean(times)

                except:
                    print('exiting in inner loop')
                    torch.cuda.empty_cache()
                    res[(input_size, output_size)] = np.nan

    except:
        print('exiting in outer loop')
        torch.cuda.empty_cache()
        res[input_size] = np.nan

with open('gpu_timings_numpy.pkl', 'wb') as f:
    pickle.dump(res, f)


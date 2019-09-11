import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
results_dir = "./"

cpu_time = np.load(results_dir+"cpu_time"+".npy") 
gpu_time = np.load(results_dir+"gpu_time"+".npy")
gpu_energy = np.load(results_dir+"gpu_energy"+".npy")

cpu_loadingtime = np.load(results_dir+"cpu_loadingtime"+".npy")
gpu_loadingtime = np.load(results_dir+"gpu_loadingtime"+".npy")
gpu_loadingenergy = np.load(results_dir+"gpu_loadingenergy"+".npy")


d_list = [100,320,1000,3200,10000,32000] # Their dimension
p_list = [100,320,1000,3200,10000,32000] # Their targeted dimension
repetitions = 600
batchsize = 100

def savfig(Z,log=False,unit="",quantity="",file = "fig.png"):
    fig,ax = plt.subplots()
    if log:
        plt.pcolor(Z,norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()))
    else:
        plt.pcolor(Z)
    ax.set_yticks(np.arange(len(d_list))+0.5)
    ax.set_yticklabels(d_list)
    ax.set_xticks(np.arange(len(p_list))+0.5)
    ax.set_xticklabels(p_list)
    plt.xlabel("Input dimension")
    plt.ylabel("Output dimension")
    clb = plt.colorbar() 
    clb.ax.set_title(unit)
    plt.title(quantity+" for "+str(repetitions)+" projections of matrices\n of "+str(batchsize)+" batch points.")
    plt.savefig(file)

savfig(gpu_energy*repetitions,True,"Joule","Energy","E_gpu.png")
savfig(gpu_time*repetitions,False,"Second","Duration","t_gpu.png")
savfig(gpu_energy/gpu_time,False,"Watt","Average power","P_gpu.png")


savfig(gpu_loadingenergy*repetitions,True,"Joule","Loading energy","E_lgpu.png")
savfig(gpu_loadingtime*repetitions,False,"Second","Loading time","t_lgpu.png")
savfig(gpu_loadingenergy/gpu_loadingtime,False,"Watt","Loading power","P_lgpu.png")


"""
plt.imshow(cpu_loadingtime)
plt.title("cpu_loadingtime")
plt.show()
plt.imshow(gpu_loadingtime)
plt.title("gpu_loadingtime")
plt.show()
plt.imshow(gpu_loadingenergy)
plt.title("gpu_loadingenergy")
plt.show()
plt.imshow(cpu_time)
plt.title("cpu_time")
plt.show()
plt.imshow(gpu_time)
plt.title("gpu_time")
plt.show()
plt.imshow(gpu_energy)
plt.title("gpu_energy")
plt.show()
"""
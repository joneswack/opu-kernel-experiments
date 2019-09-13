import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
results_dir = "./"
fig_dir = "./figures/"

OPU_POWER = 30


#cpu_time = np.load(results_dir+"cpu_time"+".npy") 
#cpu_loadingtime = np.load(results_dir+"cpu_loadingtime"+".npy")

gpu_time = np.load(results_dir+"gpu_time"+".npy")
gpu_energy = np.load(results_dir+"gpu_energy"+".npy")
gpu_power = gpu_energy/gpu_time

gpu_loadingtime = np.load(results_dir+"gpu_loadingtime"+".npy")
gpu_loadingenergy = np.load(results_dir+"gpu_loadingenergy"+".npy")
gpu_loadingpower = gpu_loadingenergy/gpu_loadingtime

opu_time = gpu_time # TODO
opu_power = OPU_POWER*np.ones_like(opu_time)
opu_energy = opu_time*opu_power

opu_loadingtime = gpu_time # TODO
opu_loadingpower = OPU_POWER*np.ones_like(opu_loadingtime)
opu_loadingenergy = opu_loadingtime*opu_loadingpower


d_list = [100,320,1000,3200,10000,32000] # Their dimension
p_list = [100,320,1000,3200,10000,32000] # Their targeted dimension
repetitions = 80
batchsize = 1000

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
    plt.title(quantity+" for matrix projection\n of "+str(batchsize)+" batch points (estimated over "+str(repetitions)+" repetitions).")
    plt.savefig(fig_dir+file)



def savfig_ratio(Z,log=False,unit="",quantity="",file = "fig.png"):
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
    plt.title(quantity+" for matrix projection\n of "+str(batchsize)+" batch points (estimated over "+str(repetitions)+" repetitions)")
    plt.savefig(fig_dir+file)


savfig(gpu_energy,True,"Joule","GPU energy","energy_GPU.png")
savfig(gpu_time,False,"Second","GPU computation time","computation_time_GPU.png")
savfig(gpu_power,False,"Watt","GPU average power","average_power_GPU.png")


savfig(opu_energy,True,"Joule","OPU energy","energy_OPU.png")
savfig(opu_time,False,"Second","OPU computation time","computation_time_OPU.png")
savfig(opu_power,False,"Watt","OPU average power","average_power_OPU.png")




savfig(gpu_loadingenergy,False,"Joule","GPU loading energy","loading_energy_GPU.png")
savfig(gpu_loadingtime,False,"Second","GPU loading time","loading_time_GPU.png")
savfig(gpu_loadingpower,False,"Watt","GPU loading power","loading_power_GPU.png")


savfig(opu_loadingenergy,True,"Joule","OPU loading energy","loading_energy_OPU.png")
savfig(opu_loadingtime,False,"Second","OPU loading time","loading_time_OPU.png")
savfig(opu_loadingpower,False,"Watt","OPU loading power","loading_power_OPU.png")


## Ratios
savfig_ratio(gpu_energy/opu_energy,False,"","OPU energy 'saving-up'","energy_ratio.png")
savfig_ratio(gpu_time/opu_time,False,"","OPU speed-up","time_ratio.png")
savfig_ratio(gpu_power/opu_power,False,"","GPU average power over OPU power","average_power_ratio.png")

savfig_ratio(gpu_loadingenergy/opu_loadingenergy,False,"","GPU loading energy over OPU loading energy","loading_energy_ratio.png")
savfig_ratio(gpu_loadingtime/opu_loadingtime,False,"","GPU loading time over OPU loading time","loading_time_ratio.png")
savfig_ratio(gpu_loadingpower/opu_loadingpower,False,"","GPU loading power over OPU power","loading_power_ratio.png")


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
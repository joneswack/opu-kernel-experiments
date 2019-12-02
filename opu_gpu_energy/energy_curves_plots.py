import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex = True)
import numpy as np

OPU_POWER = 30 # W
def get_opu_speed(input_size, output_size, gen='2'): # kHz
    """
    Function returning the speed in kHz at which the OPU can be used for given input and output sizes.
    We assume input_size <= 1M and output_size <= 2M (i.e. the OPU can be used).
    """
    if gen == 2:
        return 2
    elif gen == 3:
        if input_size <= 145e3 and output_size <= 81e3:
            return 8
        elif input_size <= 290e3 and output_size <= 81e3:
            return 6.3
        elif input_size <= 583e3 and output_size <= 163e3:
            return 3.8
        elif input_size <= 1e6 and output_size <= 326e3:
            return 1.9
        elif input_size <= 1e6 and output_size <= 600e3:
            return 1
        else:
            return 0.34
    else:
        raise ValueError("gen must be 2 or 3")



RESULTS_DIR = "results_energy_curves/"
n = 3000 # Number of points that have been projected

# define the parameters to be used
gen = 2
if gen == 2:
    opu_text = "Gen 2 (2 kHz)"
elif gen == 3:
    opu_text = "Gen 3 (up to 8 kHz)"

gpu_model = "NVIDIA Tesla P100-PCIE-16GB" 

# d_list = np.load(RESULTS_DIR+"dimensions"+"_bs_"+str(n)+".npy")
d_list = np.load(RESULTS_DIR+"dimensions"+"_bs_"+str(1000)+".npy")
d_list = d_list[:len(d_list) - 1]

print(len(d_list))

# load timings and compute frequencies and consumption for GPU
gpu_time = np.load(RESULTS_DIR+"gpu_time"+"_bs_"+str(n)+".npy")[:len(d_list)]
gpu_f = n/gpu_time
gpu_ener = np.load(RESULTS_DIR+"gpu_ener"+"_bs_"+str(n)+".npy")[:len(d_list)]
# compute opu frequ, time & consumption according to the dimension
# (independent of dim for gen==2)
opu_f = np.array(tuple(1000*get_opu_speed(d,d,gen) for d in d_list))
opu_time = n/opu_f
opu_ener = opu_time*OPU_POWER


n2 = 1000
# load timings and compute frequencies and consumption for GPU
gpu_time2 = np.load(RESULTS_DIR+"gpu_time"+"_bs_"+str(n2)+".npy")[:len(d_list)]
gpu_ener2 = np.load(RESULTS_DIR+"gpu_ener"+"_bs_"+str(n2)+".npy")[:len(d_list)]
# compute opu frequ, time & consumption according to the dimension
# (independent of dim for gen==2)
opu_time2 = n2/opu_f
opu_ener2 = opu_time2*OPU_POWER

# add limit values
opu_time = [opu_time[0]] + list(opu_time) + 3*[opu_time[-1]]
opu_ener = [opu_ener[0]] + list(opu_ener) + 3*[opu_ener[-1]]
opu_time2 = [opu_time2[0]] + list(opu_time2) + 3*[opu_time2[-1]]
opu_ener2 = [opu_ener2[0]] + list(opu_ener2) + 3*[opu_ener2[-1]]

gpu_time = [gpu_time[0]] + list(gpu_time)
gpu_ener = [gpu_ener[0]] + list(gpu_ener)
gpu_time2 = [gpu_time2[0]] + list(gpu_time2)
gpu_ener2 = [gpu_ener2[0]] + list(gpu_ener2)

d_list = [0] + list(d_list) + [56000, 58000, 60000]

summary = list(zip(d_list, opu_ener, gpu_ener))

for item in summary:
    print(item)


# Graphism
def my_savefig(xaxis,gpu_data,opu_data, unit, quantity,legend=False,**kwarg):
    plt.figure(num=None, dpi=300, facecolor='w', edgecolor='k') # dpi=80, 
    for i in range(len(gpu_data)):
        gd = gpu_data[i]
        od = opu_data[i]
        if i==0:
            line_g = axs[xaxis].plot(d_list[:len(gd)],gd,label="GPU")
            line_o = axs[xaxis].plot(d_list[:len(od)],od,label="OPU")
            line_m = axs[xaxis].axvline(d_list[len(gpu_ener2)-1], color='red', label="GPU Memory Limit")
        else:
            axs[xaxis].plot(d_list[:len(gd)],gd,linestyle=":",color = line_g[0].get_color())
            axs[xaxis].plot(d_list[:len(od)],od,linestyle=":",color = line_o[0].get_color())
            
    axs[xaxis].set_ylabel(quantity+"\n("+unit+")")
    axs[xaxis].set_xlabel("Projection Dimension $D$")
    axs[xaxis].grid()
    axs[xaxis].set_ylim(bottom=0)
    axs[xaxis].set_xlim(left=0, right=60000)

    #plt.yscale("log")
    #plt.xscale("log")
    # if legend:
        # axs[xaxis].legend(bbox_to_anchor=(1.04,1), loc="center")
        # axs[xaxis].legend(loc='upper left', fancybox=False, shadow=False) # , ncol=3
        # axs[xaxis].legend(loc='upper left')


plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Helvetica"
plt.rcParams['text.usetex'] = False
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10,2))
# plt.xlabel("Projection Dimension $D$")
my_savefig(0,[gpu_time,gpu_time2],[opu_time,opu_time2], "s", "Computation Time",legend=True)
my_savefig(1,[gpu_ener,gpu_ener2],[opu_ener,opu_ener2], "Joule", "Energy Consumption")
handles, labels = axs[0].get_legend_handles_labels()
# legend = plt.figlegend(handles=handles, labels=labels, loc='upper center', ncol=3, bbox_to_anchor = [0.5, 0.5], bbox_transform = plt.gcf().transFigure)
# legend = plt.figlegend( handles, labels, loc = 'upper center', ncol=3, labelspacing=0. )
legend = fig.legend(handles, labels, loc='upper center', ncol=3)
fig.subplots_adjust(wspace=0.5)
plt.tight_layout()



fig.savefig('curve.pdf', format="pdf",
    transparent=False, bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0.2,
    metadata={
        'Author': 'S. Marmin',
        'Title': 'GPU and OPU time computation and energy consumption'})




#plt.plot(d_list,gpu_f/1000)
#plt.scatter(d_list,gpu_f/1000)
#plt.plot(d_list,opu_f/1000)

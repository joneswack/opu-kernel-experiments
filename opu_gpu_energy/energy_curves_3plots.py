import matplotlib.pyplot as plt
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
n1 = 20
n2 = 1000 # Number of points that have been projected
n3 = 3000 # Number of points that have been projected

# define the parameters to be used
gen = 2
if gen == 2:
    opu_text = "Gen 2 (2 kHz)"
elif gen == 3:
    opu_text = "Gen 3 (up to 8 kHz)"

gpu_model = "NVIDIA Tesla P100-PCIE-16GB" 

d_list = np.load(RESULTS_DIR+"dimensions"+"_bs_"+str(n1)+".npy")

# load timings and compute frequencies and consumption for GPU
gpu_time1 = np.load(RESULTS_DIR+"gpu_time"+"_bs_"+str(n1)+".npy")
gpu_f1 = n1/gpu_time1
gpu_ener1 = np.load(RESULTS_DIR+"gpu_ener"+"_bs_"+str(n1)+".npy")
gpu_time2 = np.load(RESULTS_DIR+"gpu_time"+"_bs_"+str(n2)+".npy")
gpu_f2 = n2/gpu_time2
gpu_ener2 = np.load(RESULTS_DIR+"gpu_ener"+"_bs_"+str(n2)+".npy")
gpu_time3 = np.load(RESULTS_DIR+"gpu_time"+"_bs_"+str(n3)+".npy")
gpu_f3 = n3/gpu_time3
gpu_ener3 = np.load(RESULTS_DIR+"gpu_ener"+"_bs_"+str(n3)+".npy")
# compute opu frequ, time & consumption according to the dimension
# (independent of dim for gen==2)
opu_f = np.array(tuple(1000*get_opu_speed(d,d,gen) for d in d_list))
opu_time1 = n1/opu_f
opu_ener1 = opu_time1*OPU_POWER
opu_time2 = n2/opu_f
opu_ener2 = opu_time2*OPU_POWER
opu_time3 = n3/opu_f
opu_ener3 = opu_time3*OPU_POWER

# Graphism
def my_savefig(gpu_data1,gpu_data2,gpu_data3,opu_data1,opu_data2,opu_data3, unit, quantity,legend=False,**kwarg):
    plt.figure(num=None, figsize=(6, 2), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(d_list,gpu_data1,label="GPU, n = "+str(n1),c="blue",linestyle=":")
    plt.plot(d_list,gpu_data2,label="GPU, n = "+str(n2),c="blue",linestyle="-.")
    plt.plot(d_list,gpu_data3,label="GPU, n = "+str(n3),c="blue",linestyle="--")
    plt.plot(d_list,opu_data1,label="OPU, n = "+str(n1),c="red",linestyle=":")
    plt.plot(d_list,opu_data2,label="OPU, n = "+str(n2),c="red",linestyle="-.")
    plt.plot(d_list,opu_data3,label="OPU, n = "+str(n3),c="red",linestyle="--")
    plt.xlabel("Projection dimension")
    plt.ylabel(quantity+"\n("+unit+")")
    plt.tight_layout()
    plt.yscale("log")
    plt.xscale("log")
    if legend:
        plt.legend()
    plt.savefig('curve_'+quantity.lower().replace(" ", "_")+'.pdf', format="pdf",
        transparent=False,bbox_inches=None, pad_inches=0.1,
        metadata={
            'Author': 'S. Marmin',
            'Title': 'GPU and OPU '+quantity.lower()
            },**kwarg)
    plt.close()


my_savefig(gpu_ener1,gpu_ener2,gpu_ener3,opu_ener1,opu_ener2,opu_ener3, "Joule", "Energy consumption",legend=True)
my_savefig(gpu_time1,gpu_time2,gpu_time3,opu_time1,opu_time2,opu_time3, "s", "Computation time")
#my_savefig(gpu_f/1000,opu_f/1000, "kHz", "Multiplication frequency")


#plt.plot(d_list,gpu_f/1000)
#plt.scatter(d_list,gpu_f/1000)
#plt.plot(d_list,opu_f/1000)

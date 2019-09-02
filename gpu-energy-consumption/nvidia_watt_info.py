from subprocess import Popen
from time import time


command = 'python my_script.py'

process = subprocess.Popen(command, shell=True)
time.sleep(5)
process.terminate()

class EnergyMonitor():
    def __init__(self, output_file=None,keep_data=False): # TODO keep_data
        if output_file is None:
            output_file = "energy_comsumption_"+str(time)+".csv"
        self.output_file = output_file
        self.monitoring_period = 10 # in seconds
        self.nvidia_command = self._setup_command()
        self._setup_command()
        self.process = subprocess.Popen(self.nvidia_command, shell=True)
        print("Start monitoring GPUs energy consumption in background.")

    def _setup_command(self):
        com = "nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l "+str(self.monitoring_period)" > "+self.output_file
        self.nvidia_command = com

    def energy():
        newpd = pd.DataFrame(pd.read_csv(self.output_file, typ='series')).T
        gpu_names = list(unique(newpd["index"]))
        ngpu = len(gpu_names)
        energies = [] # list of floats
        for gpu in gpu_names:
            dataframe = newpd[newpd["index"]==gpu]
            t_str = dataframe["timestamp"]
            p_str = dataframe["power.draw"]
            t = numpy(t_str)
            p = numpy(p_str)
            energies += [integrate(t,p)]
        timobj = t[-1]
        return Energy(timobj,gpu,energies)
        

class Energy(t,gpus,energies_list):
    self.t = t
    self.gpus = gpus
    self.enerlist = energies_list

    def __repr__(self):
        return self.t.__repr__()+"\n"+("gpu "self.gpus[i]+": "+str(self.enerlist[i])+" J" for i in len(self.gpus))
    def __str__(self):
        return "member of Test"

def integrate(t,p): # trapezoidal integration
    # if `p` is in Watt, and `x` in seconds, output is in Joule. # TODO
    return sum((t[1:]-t[:-1])*(p[1:]+p[:-1])*.5)


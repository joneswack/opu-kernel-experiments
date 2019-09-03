from subprocess import Popen, check_call
from time import time, sleep
import os
from os import remove
import pandas as pd
from datetime import datetime, timedelta
from numpy import empty, array, sum as np_sum, float32, timedelta64,Inf
from io import StringIO

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# shell command whose periodic line results (by appending -l [int])
# will be converted to a dataframe by EnergyMonitor.to_pandas()
# power must be in Watt.
GPU_COMMAND = ["nvidia-smi", "--query-gpu=index,timestamp,power.draw", "--format=csv"]


##### Auxiliary formating functions
def make_datetime(text):
    return datetime.strptime(text.strip('" '),'%Y/%m/%d %H:%M:%S.%f')
    
def make_float(text):
    if text[-1] == "W":
        text = text[:-1]
    return float(text.strip('" '))
    
def make_str(text):
    return text.strip('" ')

def wait_for_nonempty(file):
    period = .2
    echeance = 15
    allesKlar = False
    for i in range(int(echeance/period)+1):
        if os.path.getsize(file) > 0:
            allesKlar = True
            break
        else:
            sleep(period)
    if not allesKlar:
        raise IOError("File "+str(file)+" stays empty, but should contain output from "+GPU_COMMAND+", launched in a parallel subprocess in .energy().")


##### Main (and unique) class

class EnergyMonitor():
    def __init__(self, output_file=None,period=10,keep_data=False):
        # period is the integration interval in seconds.
        # ouput_file is temporary by default (i.e. will be deleted)
        # keep_data is to keep this raw data.

        if output_file is None:
            tempstr = "" if keep_data else "temp_"
            output_file = "energy_comsumption_"+tempstr+str(time())+".csv"
        self._output_file = output_file
        self._period = period # in seconds
        self.keep_data = keep_data
        cmd = os.popen(" ".join(GPU_COMMAND)).read()
        f = StringIO(cmd)
        df = self.to_pandas(f)
        self._gpus = list(df["gpu"])
        powers = list(df["power"])
        self._ngpus = len(self._gpus)
        print("Detected GPU(s):",end = " ")
        st = ""
        for gpu,p in zip(self._gpus,powers):
            st+="GPU "+gpu+" ("+str(p)+" W), "
        print(st[:-2],end=".\n")
        self._process = None
        
    def __del__(self):
        if self._process is not None:
            self._process.terminate()
        print("Stop monitoring GPUs.")
        if not self.keep_data:
            try:
                remove(self._output_file)
            except IOError:
                pass
        
    def to_pandas(self,f=None):
        if f is None:
            f = self._output_file
        df = pd.DataFrame(pd.read_csv(f,
                names=["gpu", "time", "power"],
                skiprows=[0],
                converters = {"gpu" : make_str,
                              "time":make_datetime,
                                    'power' : make_float}
                                       ))
        return df
    
    
    def to_numpy(self):
        newpd = self.to_pandas()
        gpu_names = list(pd.unique(newpd["gpu"]))
        ngpu = len(gpu_names)
        df1 = newpd[newpd["gpu"]==gpu_names[0]]
        n = len(df1.index)
        t = empty((ngpu,n),dtype='datetime64[ms]')
        p = empty((ngpu,n))
        t[0,:] = df1["time"]
        p[0,:] = df1["power"]
        base = datetime(2000, 1, 1)
        for i in range(1,ngpu):
            dfi =  newpd[newpd["gpu"]==gpu_names[i]]
            t[i,:] = dfi["time"]
            p[i,:] = dfi["power"]
        return t, p, gpu_names
    
    def energy(self):
        if self._process is None:
            print("Starting GPU energy monitoring in background...",end = ' ')
            self._process = Popen(GPU_COMMAND+["-l"]+[str(self._period)],stdout=open(self._output_file,"a"))
            wait_for_nonempty(self._output_file)
            print("OK.")
            

        t, p, gpus = self.to_numpy()
        e = list(integrate(t,p))
        return Energy(t[0,-1],dict(zip(gpus,e)))
    

        
class Energy():
    def __init__(self,t,energies_dict):
        self.t = t
        self.e = energies_dict
        self.dt = isinstance(t,timedelta64)

    def __repr__(self):
        return "Energy("+repr(self.t)+",\n"+repr(self.e)+")"
    def __str__(self):
        res = " GPU | Energy (J)"
        if self.dt:
            delta = self.t.item().total_seconds()
            if delta>10**(-50):
                res += " | Avg. pow (W)"
        for gpu, ene in self.e.items():
            res += "\n {:>3} ".format(gpu)
            res += '| {:10.2f} '.format(ene)
            if self.dt and delta>10**(-50):
                res += '|  {:11.2f}'.format(float(ene)/delta)
        if self.dt:
            res +="\n over a period of "+str(self.t)+"."
        else:
            res +="\n at "+str(self.t)+"."
        return res

    def __sub__(self,other):
        dE = (e1-e2 for e1,e2 in zip(self.e.values(),other.e.values()))
        return Energy(self.t-other.t,
                           dict(zip(self.e.keys(),dE)))
    
    def __add__(self, other):
        E = (e1+e2 for e1,e2 in zip(self.e.values(),other.e.values()))
        return Energy(self.t+other.t,
                           dict(zip(self.e.keys(),E)))
    __radd__ = __add__
    
    def __abs__(self):
        E = (abs(e1) for e1 in self.e.values())
        return Energy(abs(self.t),dict(zip(self.e.keys(),E)))
    
    def item(self,gpu):
        return self.e[gpu]

    # TODO more fine comparisons involving enerdict and a __hash__?
    def __eq__(self, other):
        return self.t==other.t
    def __le__(self, other):
        return self.t<=other.t
    def __lt__(self, other):
        return self.t< other.t
    def __ge__(self, other):
        return self.t>=other.t
    def __gt__(self, other):
        return self.t> other.t
    
    
def integrate(t,p):# trapezoidal integration
    # if `p` is in Watt, and `t` in miliseconds, output is in Joule.
    return np_sum((t[...,1:]-t[...,:-1]).astype(float32)/1000*(p[...,1:]+p[...,:-1])*.5,axis=-1)


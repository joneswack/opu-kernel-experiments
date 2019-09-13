from subprocess import Popen, check_call
from time import time, sleep
import os
from os import remove
import pandas as pd
from datetime import datetime, timedelta
from numpy import empty, array, sum as np_sum, float32, timedelta64,Inf, datetime64, zeros
from io import StringIO
from warnings import warn

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
        if self._ngpus>0:
            print("Detected GPU(s):",end = " ")
            st = ""
            for gpu,p in zip(self._gpus,powers):
                st+="GPU "+gpu+" ("+str(p)+" W), "
            print(st[:-2],end=".\n")
        else:
            print("No GPU detected.")
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
    
    
    def analized_pandas(self):
        same_number_of_measurements = True
        newpd = self.to_pandas()
        gpu_names = list(pd.unique(newpd["gpu"]))
        ngpus = len(gpu_names)
        if ngpus != self._ngpus:
            warn("The number of GPUs found in self.pandas() ("+str(ngpus)+") is different from the initial believe ("+str(self._ngpus)+").")
        nmax = 0
        dfs = []
        for i in range(ngpus):
            dfi = newpd[newpd["gpu"]==gpu_names[i]]
            n = len(dfi.index)
            if i>0 and n!=nmax:
                st = "The number of measurements ("+str(n)+") for GPU "+gpu_names[i]+" is different from the ones we have seen so far ("+str(nmax)+") while parsing self.pandas() with GPU(s) "
                for j in range(i):
                    st+= gpu_names[j]+", "
                st= st[:-2]+"."
                warn(st)
                same_number_of_measurements = False
            if n>nmax:
                nmax = n
            dfs += [dfi]
        return same_number_of_measurements, nmax, ngpus, dfs, gpu_names
        

    def to_numpy(self):
        if self._ngpus==0:
            return empty((0,0),dtype='datetime64[ms]'),empty((0,0)),[]
        ntries = 6
        delay  = .1

        for j in range(ntries):
            restart=False
            same, n, ngpus, dfs, gpu_names = self.analized_pandas()
            if not same or ngpus!=self._ngpus:
                if j<ntries-1:
                    print("Waiting "+str(delay)+" s to see if the subprocess is writting something... (doesn't affect the measurments).")
                    restart=True
                    sleep(delay)
                else:
                    print("Waiting did not pay off. Reuse last measurments.")    
            elif j>0:
                print("Waiting paid off.")
            if not restart:
                break
        
        t = empty((ngpus,n),dtype='datetime64[ms]')
        p = empty((ngpus,n))
        for i in range(ngpus):
            t_p = array(dfs[i]["time"])
            p_p = array(dfs[i]["power"])
            t[i,:len(t_p)] = t_p
            t[i,len(t_p):] = t_p[-1]
            p[i,:len(p_p)] = p_p
            p[i,len(p_p):] = p_p[-1]
    
        return t, p, gpu_names
    
    def energy(self):
        if self._process is None and self._ngpus>0:
            print("Starting GPU energy monitoring in background...",end = ' ')
            self._process = Popen(GPU_COMMAND+["-l"]+[str(self._period)],stdout=open(self._output_file,"a"))
            wait_for_nonempty(self._output_file)
            print("OK.")
            

        t, p, gpus = self.to_numpy()
        now_micro = datetime.now()
        now = datetime64(now_micro).astype('datetime64[ms]')
        
        if t.shape[-1]>0 and self._ngpus>0:
            now_got = t[0,-1]
            if now_got>now:
                raise NotImplementedError("Something is wrong. This error should never appear as the dates in "+str(self._output_file)+" should always be before self.energy() call ("+str(now)+").")
            p_chec = self._period +.5
            diff = (now-now_got).item().total_seconds()
            if diff>p_chec: # check if we work on recent value
                warn("Last power evaluations was more than "+str(diff)+" seconds ago. Check if the background process "+str(self._process)+" is still writing in "+str(self._output_file)+".")
        e = list(integrate(now,t,p))
        return Energy(now,dict(zip(gpus,e)))
    

        
class Energy():
    def __init__(self,t,energies_dict):
        self.t = t
        self.e = energies_dict
        self.dt = isinstance(t,timedelta64)
        self.ngpus = len(energies_dict)

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

    def duration(self):
        if self.dt:
            return self.t.item().total_seconds()
        else:
            raise ValueError("This Energy object is not a 'delta'.")

    def select_gpu(self,gpu):
        return Energy(self.t,{gpu:self.e[gpu]})

    def consumption(self,gpu=None): # return float.
        if not self.dt:
            raise ValueError("This Energy object is not a difference between to time points. This is the energy since the first call EnergyMonitor.energy().")
        if self.ngpus==0:
            return 0. # No GPUs means no consumption
        if gpu is None:
            if self.ngpus==1:
                return list(self.e.values())[0]
            else:
                raise ValueError("More than one GPU, but 'gpu' is not specified. Chose between gpu = "+str(self.e.keys())+" (must be str type, not int).")
        else:
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
    
    
def integrate(now,t,p):# trapezoidal integration
    # if `p` is in Watt, and `t` in miliseconds, output is in Joule.
    second = timedelta64(1, 's') # for conversion to float
    if t.shape[-1]==0:
        return zeros((*t.shape[:-1],0),dtype=float32)
    I = np_sum((t[...,1:]-t[...,:-1])/second*(p[...,1:]+p[...,:-1])*.5,axis=-1)
    I += (now-t[...,-1])/second*p[...,-1]
    return I






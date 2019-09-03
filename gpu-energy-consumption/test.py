from nvidia_watt_info import EnergyMonitor
import time
e = EnergyMonitor(period = 3)

ener1 = e.energy()
time.sleep(5)
ener2 = e.energy()
print(ener1)
print(ener2)

print(repr(ener1))
print(repr(ener2))

print(repr(ener2-ener1))
print((ener2-ener1))
print((ener2-ener1).item("2"))


#print(p)
#import matplotlib
#matplotlib.rcParams['timezone'] = 'Europe/Paris'
#import matplotlib.pyplot as plt

#plt.plot(t[0,:],p[0,:])
#plt.savefig("figure.png")
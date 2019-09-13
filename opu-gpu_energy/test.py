from power_info import EnergyMonitor
import time
e = EnergyMonitor(period = 3)

ener1 = e.energy()
time.sleep(.5)
ener2 = e.energy()
print(ener1)
print(ener2)

print(repr(ener1))
print(repr(ener2))

print(repr(ener2-ener1))
print((ener2-ener1))
print((ener2-ener1).consumption(gpu="2"))



import matplotlib
matplotlib.rcParams['timezone'] = 'Europe/Paris'
import matplotlib.pyplot as plt

t,p,gpu = e.to_numpy()

for i in range(len(gpu)):
    plt.plot(t[i,:],p[i,:],label="GPU "+gpu[i])
plt.legend()
plt.savefig("figure.png")
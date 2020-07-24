import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

#time = range(0,140000) # For Byrce's paper
time = range(0,12000)
P = np.zeros(len(time))
t = np.zeros(len(time))
#Pmax = 0.002		# For Bryce's paper
Pmax = 0.001
Period = 3000
#Period = 13863.53	# For Bryce's paper


t1 = Period/2.6;
t2 = Period/1.95;
t3 = Period/1.69;
t4 = Period/1.52;
t5 = Period/1.25;
t6 = Period/1.03;

m1 = 0.28/3.68;
m2 = 2.622806/3.68;
idx = 0
for Time in time:
    TimeMod = (int(Time) + int(t1)) % int(Period)
    
    if (TimeMod < t1): 
        Pin = 0
    elif (TimeMod >= t1 and TimeMod < t2):
        Pin = Pmax*m1*(TimeMod-t1)/(t2-t1);
    elif (TimeMod >= t2 and TimeMod < t3):
        Pin = -Pmax*m1*(TimeMod-t3)/(t3-t2);
    elif (TimeMod >= t3 and TimeMod < t4):
        Pin = Pmax*m2*(TimeMod-t3)/(t4-t3);
    elif (TimeMod >= t4 and TimeMod < t6):
        Pin = (Pmax-m2*Pmax)/((t5-t4)*(t5-t6))*(TimeMod-t4)*(TimeMod-t6)+m2*Pmax;
    elif (TimeMod >= t6 and TimeMod <= Period):
        Pin = -Pmax*m2*(TimeMod-Period)/(Period-t6)
    P[idx] = Pin
    t[idx] = Time
    idx +=1

#Read and interp the alternate pressure profile
alternate = pd.read_csv('Alternate_PressureIn.csv', header=None, names=['t0', 'p0'])
alternate.t0 = Period * alternate.t0.values / max(alternate.t0.values)
alternate.p0 = alternate.p0 - min(alternate.p0)
alternate.p0 = Pmax   * alternate.p0.values / max(alternate.p0.values)

from scipy import interpolate
Pinterp = interpolate.interp1d(alternate.t0.values, alternate.p0.values)
myPressures = np.zeros(len(time))
idx = 0
for Time in time:
    myPressures[idx] = Pinterp( Time % int(Period))
    idx +=1


print(alternate.head())

np.savetxt('PressureIn.csv', np.column_stack((t.flatten(), P.flatten())), delimiter=',', fmt='%e',header="\"t\",\"P\"", comments='')
np.savetxt('PressureIn_alternate.csv', np.column_stack((t.flatten(), myPressures.flatten())), delimiter=',', fmt='%e',header="\"t\",\"P\"", comments='')

#np.savetxt('tikzData.csv', np.column_stack((t.flatten(), P.flatten())), delimiter=',', fmt='%.8f')
#np.savetxt('tikzData2.csv', np.column_stack((t.flatten(), myPressures.flatten())), delimiter=',', fmt='%.8f')

plt.plot(t,P, 'b')
#plt.plot(alternate.t0.values, alternate.p0.values)
plt.plot(time, myPressures, 'r')
plt.legend(['Bryces Paper', 'Alternate Pressure'])
plt.grid('both')
plt.show()

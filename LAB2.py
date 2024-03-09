# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 13:25:20 2022

@author: adria
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

Id=np.zeros(22)
Id[0]=0.001
Id[1]=0.002
Id[2]=0.003
Id[3]=0.004
Id[4]=0.005
Id[5]=0.006
Id[6]=0.007
Id[7]=0.008
Id[8]=0.009
Id[9]=0.01
Id[10]=0.011
Id[11]=0.013
Id[12]=0.015
Id[13]=0.020
Id[14]=0.023
Id[15]=0.028
Id[16]=0.035
Id[17]=0.040
Id[18]=0.050
Id[19]=0.070
Id[20]=0.098
Id[21]=0.116
Id=Id*1000


Y = np.log(Id)

Vd=np.zeros(22)
Vd[0]=0.428
Vd[1]=0.477
Vd[2]=0.504
Vd[3]=0.523
Vd[4]=0.536
Vd[5]=0.536
Vd[6]=0.562
Vd[7]=0.572
Vd[8]=0.581
Vd[9]=0.589
Vd[10]=0.596
Vd[11]=0.609 
Vd[12]=0.622
Vd[13]=0.646
Vd[14]=0.657
Vd[15]=0.675
Vd[16]=0.695
Vd[17]=0.705
Vd[18]=0.726
Vd[19]=0.755
Vd[20]=0.782
Vd[21]=0.8
Vd=Vd*1000



c11=0
for i in range (0,22):
    c11 = c11 + 22*Y[i]*Vd[i]

c14=0
for i in range (0,22):
    c14 = c14 + Vd[i]

c14 = c14**2

sVd=0
sY=0
syx=0
s2Vd=0
for i in range(0, 22):
    sVd = sVd + Vd[i]
    sY=sY+Y[i]
    s2Vd= s2Vd+Vd[i]**2
    syx=syx+Y[i]*Vd[i]
c11=22*syx
c12=sVd*sY
c13=22*s2Vd
c14=sVd**2

    
c1 = (c11-c12)/(c13-c14)
n=1/(c1*25.8)



c0sVd=0
c0sY=0
c0syx=0
c0s2Vd=0
for i in range(0, 22):
    c0sVd = c0sVd + Vd[i]
    c0sY=c0sY+Y[i]
    c0syx=c0syx+Y[i]*Vd[i]
    c0s2Vd=c0s2Vd+(Vd[i]**2)
c01=c0sY*c0s2Vd    
c02=c0sVd*c0syx
c03=22*c0s2Vd
c04 = c0sVd**2

c0 = (c01-c02)/(c03-c04)
Vdt = np.linspace(0, 800, 1000)
Is=np.exp(c0)
Y2=np.log10(Id)

y = Is*(np.exp(Vdt/(n*25.8))-1)
#plt.plot(Vd,Id,"g.")
#plt.plot(Vdt,y,"r-")
plt.plot(Vd,Y2,"b.")
plt.xlabel('Vd(mV)')
plt.ylabel('Log(Id(mA))')
plt.axis([0,1000,0,5])

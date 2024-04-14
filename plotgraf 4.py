# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:16:06 2022

@author: adria
"""

import numpy as np
import matplotlib.pyplot as plt
p=2
t1=[]
ch1=[]
t2=[]
ch2=[]



with open('fourier sinal 3/T0000MTHb.csv') as csvfile:
    spamreader = np.genfromtxt(csvfile, delimiter=',')
    cont=0
    for row in spamreader:
        if cont%p==0:
            a=[]
            for i in row:
                a.append(i)
            t1.append(a[0])
            ch1.append(a[1])
            #t2.append(a[3])
            #ch2.append(a[4])
        for i in row:
            a.append(i)
        t1.append(a[0])
        ch1.append(a[1])
        cont=cont+1




plt.plot(t1,ch1,"-")
#plt.plot(t2,ch2,"b-")
#
#plt.plot(t48,ch148,'b-')
#plt.plot(tsk1,ch2sk1,"r-")


#plt.plot(t48,ch148,"b-")

#plt.plot(t243,ch1243,"y-")
#plt.plot(tsk1,ch1sk1,'r-')

p#lt.legend(['sinal de entrada','1N148','1N4007','BA243', 'SK1/12' ])

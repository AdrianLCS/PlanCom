import numpy as np
n=1.4
p=0.286
bc=7*(10**(-11))
K=1.381*(10**(-23))
Tf=1400
lambd=0.63*(10**(-6))
Yr=(8*(np.pi**3)/(3*(lambd**4)))*(n**8)*(p**2)*bc*K*Tf

l=np.exp(-Yr)

print(l)

a=10*np.log10(-1/l)
print(a)


import numpy as np
def ikegami_model(h, hr, f, w=22.5, lr=2, th=np.pi / 2):  # f em MHz
    # h-predio, hr-receptor em m e f em MHz
    # th é o angulo entre linha de visada e a rua dado em radianos pois é argumento de np.sin()
    # w é a largura da rua
    l = - 5.8 - 10 * np.log10(1 + (3 / (lr ** 2))) - 10 * np.log10(w) + 20 * np.log10(h - hr) + 10 * np.log10(
        np.sin(th)) + 10 * np.log10(f)
    return l
def min_alt_ikegami( f, w=22.5, lr=2, th=np.pi / 2):
    L = -1
    min_alt = 0
    while (L < 0):
        min_alt = min_alt + 0.1
        L = ikegami_model(min_alt,0 , f, w, lr, th)
    return min_alt


w=22.5
h=5.3
hr=1.7
f=49
a=ikegami_model(h, hr, f, w)
b=min_alt_ikegami(800)
print(b)
print(a)
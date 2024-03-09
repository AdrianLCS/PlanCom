import numpy as np

def perda(AR, Ga,Gb, tau=0,teta=90, polariazacao='linear'):#INTRODUCTION TO RF PROPAGATION, Pag 34 ou 54 do pdf/ pag 58 ou 77 do pdf
    tetarad = teta * np.pi / 180
    taurad = tau * np.pi / 180 #diferença de polariazação entre a onda incidente e a antena receptora
    if polariazacao == 'circular':
        F = ((1+AR**2*Ga/Gb)*(1+AR**2)+4*AR**2*((Ga/Gb)**0.5)+(1-(AR**2)*(Ga/Gb))*(1-(AR**2)))*np.cos(2*tetarad)/(2*(1+(AR**2)*(Ga/Gb))*(1+(AR**2)))
    else:
        F = np.cos(taurad)**2

    return F

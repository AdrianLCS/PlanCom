from numpy import np
import math
c = 299792458  # m/s

def link_margin(eirp, lpath, gr, thrx):
    # Potencia de tranmissaao isotropica = Pt*Gt*Perdadocabo*perdaaleatoria, perda do caminho, ganho de Rx e Sensibilidade
    margem = eirp - lpath + gr - thrx
    return margem


def friis_free_space_loss_com_ganho(gt, gr, comprimento_de_onda, d):  # gt=direcionalidade*eficiencia
    L = gt * gr * ((comprimento_de_onda / (4 * np.pi * d)) ** 2)
    return L  # Isso na verdade é um ganho ao invés de perda


def friis_free_space_loss_com_ganho_db(gt_db, gr_db, comprimento_de_onda, d):  # gt=direcionalidade*eficiencia
    L_db = -gt_db - gr_db - 20 * np.log10(comprimento_de_onda) + 20 * np.log10(d) + 22
    return L_db # Esse valor é de fato uma perda


def friis_free_space_loss(comprimento_de_onda, d):  # gt=direcionalidade*eficiencia
    L = ((comprimento_de_onda / (4 * np.pi * d)) ** 2)
    return L  # Isso na verdade é um ganho ao invés de perda


def friis_free_space_loss_db(f, d):  # gt=direcionalidade*eficiencia
    comprimento_de_onda=c/(f*1000000)
    L_db = 20 * np.log10(comprimento_de_onda) + 20 * np.log10(d) + 22
    return L_db # Esse valor é de fato uma perda

def k_raioEquivalente (h):
    H=7 #km
    r=6370 #km
    req = 1/((-312*np.exp(-h/H)*(10**(-6))/H)+(1/r))  #Pag 133 do Pdf, h em km
    k=r/req
    return k


def distancia_de_horizonte(h,k):
    raio = 6378137
    d=(2*k*h*raio)**0.5
    return d

#Ocorre apenas em áreas umidas e costeiras, ou planas. Mas pode ser usado para dar uma margem de segurança juntamento com chuva
def Probab_atenuacao_mutltPath_atm(A, f, nd1, d, hr, he): #hr-altura antena recp  he- altura antena emiss
    #pag 137 pdf
    K = 10**(-4.2-0.0029*nd1)
    hl = min(hr, he)
    ep = abs(hr-he)/d
    p = K*(d**3)*((1+abs(ep))**(-1.2))*(10**(0.033*f-0.001*hl-(A/10)))
    return p

listaFreq=[range(1,10)]
listaAten=[0.004,0.006, 0.007, 0.0075, 0.008, 0.009, 0.01, 0.012, 0.014, 0.016] #db/km
def atenuacao_atmosferic(f,d): #d em km e f em GHz
    if f < 1:
        freq = 1
    else:
        freq = round(f)
    i = listaFreq.index(freq)
    a = listaAten[i]
    A = a*d
    return A #db

def atenuaca_vegetacao_weiss(f,d):  #F em GHz de em metros de de 235M a 95 G
    if d <= 14:
        L= 0.45*(f**0.248)*d
    else:
        L = 1.33 * (f ** 0.248) * (d**0.588)
    return L # L em dB


def atenuaca_vegetacao_antiga_ITU(f,d):  #F em GHz e d em metros
    L= 0.2*(f**0.3)*(d**0.6)
    return L # L em dB

def itu_terain_model(h,d1,d2,f, d): #perda além do espaço livre
    #h=Negativo da atura do bloqueio em relação a linha de visada
    #d1 e d2 é a distancia de cada antena para o bloqueio em km
    # f em GHz
    F1=17.3*((d1*d2/(f*d))**0.5) #F1 em metros
    A = -20*h/F1 + 10 #A em db
    return A





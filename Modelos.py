import numpy as np
import math

c = 299792458  # m/s

def raio_fresnel(n, d1, d2, f):
    # f em Mhertz
    return (n * (c / (f*1000000)) * d1 * d2 / (d1 + d2)) ** 0.5

def atenuaca_vegetacao_antiga_ITU(f, d):  # F em GHz e d em metros
    L = 0.2 * (f ** 0.3) * (d ** 0.6)
    return L  # L em dB


def friis_free_space_loss_db(f, d):  # gt=direcionalidade*eficiencia
    comprimento_de_onda = c / (f * 1000000)
    L_db = -20 * np.log10(comprimento_de_onda) + 20 * np.log10(d) + 22
    return L_db  # Esse valor é de fato uma perda


def dhs(s, Dh):
    D = 50000
    return Dh * (1 - 0.8 * np.exp(-s / D))


def sigmahs(s, Dh):
    Dhs = dhs(s, Dh)
    return 0.78 * Dhs * np.exp(-((Dh / 16) ** (1 / 4)))


def fn(v):
    if v <= 2.4:
        return 6.02 + 9.11 * v - 1.27 * (v ** 2)
    else:
        return 12.963 + 20 * np.log10(v)


def ak(v1, v2):
    return fn(v1) + fn(v2)


def vj(teta, k, dlj, s, dl):
    return (teta / 2) * (((k / np.pi) * ((dlj * (s - dl)) / (s - dl + dlj))) ** 0.5)


def adif(s, k, Dh, he1, he2, hg1, hg2, dl, tetae, Ye, dls, dl1, dl2, teta, Ar):
    C = 10  # m^2
    Q = min((k / (2 * np.pi)) * (dhs(s, Dh)), 1000) * (((he1 * he2 + C) / (hg1 * hg2 + C)) ** 0.5) + (
            dl + tetae / Ye) / s
    w = 1 / (1 + 0.1 * (Q ** 0.5))
    alpha = 4.7e-4  # m^2
    Af0 = min(15, 5 * np.log10(1 + alpha * k * hg1 * hg2 * sigmahs(dls, Dh)))
    v1 = vj(teta, k, dl1, s, dl)
    v2 = vj(teta, k, dl2, s, dl)
    adifv = (1 - w) * ak(v1, v2) + w * Ar + Af0
    return adifv


def Yj(hej, dlj):
    valor = max(1e-12, (2 * hej) / (dlj ** 2))
    return valor


def alphaj(k, Yj):
    valor = (k / Yj) ** (1 / 3)
    return valor


def kj(alphaj, Zg):
    return complex(1, 0) / (alphaj * Zg * 1j)


def xj(alphaj, Yj, dlj, Kj):
    A = 151.3
    return A * b(Kj) * alphaj * Yj * dlj


def x(alphaj, Kj, teta, x1, x2):
    A = 151.3
    return A * b(Kj) * alphaj * teta + x1 + x2


def g(x):
    return 0.05751 * x - np.log10(x)


def f(xj, Kj):
    F1 = 40 * np.log10(max(xj, 1)) - 117
    if (abs(Kj) < 10 ** (-5)) or xj * ((-np.log10(abs(Kj))) ** 3) > 450:
        F2 = F1
    else:
        F2 = (2.5 * (10 ** (-5)) * (xj ** 2) / abs(Kj)) + 20 * np.log10(abs(Kj)) - 15
    if (xj > 0) and (xj <= 200):
        return F2
    elif (xj > 200) and (xj < 2000):
        return g(xj) + 0.134 * xj * np.exp(-xj / 200) * (F1 - g(xj))
    else:
        return g(xj)


def c1(K):
    return 20  # db


def b(K):
    return 1.607 - abs(K)


def alos(k, dls, Dh, s, md, Aed, he1, he2, Zg):
    D1 = 47.7
    D2 = 10000
    w = 1 / (1 + D1 * k * Dh / max(D2, dls))
    Ad = Aed + md * s
    senfi = (he1 + he2) / (((s ** 2) + ((he1 + he2) ** 2)) ** 0.5)
    Rel = ((senfi - Zg) / (senfi + Zg)) * np.exp(-k * sigmahs(s, Dh) * senfi)
    if abs(Rel) >= max(0.5, senfi ** 0.5):
        Re = Rel
    else:
        Re = (Rel / abs(Rel)) * (senfi ** 0.5)
    deltal = 2 * k * he1 * he2 / s
    if deltal <= np.pi / 2:
        delta = deltal
    else:
        delta = np.pi - (((np.pi / 2) ** 2) / deltal)
    At = -20 * np.log10(abs(1 + Re * np.exp(delta * 1j)))

    Alos = (1 - w) * Ad + w * At
    return Alos


def rj(k, tetal, hej):
    return 2 * k * tetal * hej


def f0(D):
    if (D > 0) and (D < 10000):
        f0v = 133.4 + 0.332 * (10 ** (-3)) * D - 10 * np.log10(D)
    elif (D > 10000) and (D < 70000):
        f0v = 104.6 + 0.212 * (10 ** (-3)) * D - 2.5 * np.log10(D)
    else:
        f0v = 71.8 + 0.157 * (10 ** (-3)) * D + 5 * np.log10(D)
    return f0v


def achar_zs(elevacao):  # altura além do nivel do mar
    zs = np.mean(elevacao)
    return zs


def f2(D, Ns):
    D0 = 40000
    f2v = f0(D) - 0.1 * (Ns - 301) * np.exp(-(D / D0))
    return f2v


def h01(r, j):
    h01v = 0
    if j == 1:
        h01v = 10 * np.log10(1 + 24 * (r ** (-2)) + 25 * (r ** (-4)))
    elif j == 2:
        h01v = 10 * np.log10(1 + 45 * (r ** (-2)) + 80 * (r ** (-4)))
    elif j == 3:
        h01v = 10 * np.log10(1 + 68 * (r ** (-2)) + 177 * (r ** (-4)))
    elif j == 4:
        h01v = 10 * np.log10(1 + 80 * (r ** (-2)) + 395 * (r ** (-4)))
    elif j >= 5:
        h01v = 10 * np.log10(1 + 105 * (r ** (-2)) + 705 * (r ** (-4)))
    return h01v


def h00(r1, r2, ett):
    ett = round(ett)
    if ett == 0:
        h00v = 10 * np.log10(
            ((1 + (2 ** 0.5) / r1) ** 2) * ((1 + (2 ** 0.5) / r2) ** 2) * (r1 + r2) / (r1 + r2 + 2 * (2 ** 0.5)))
    else:
        h00v = 0.5 * (h01(r1, ett) + h01(r2, ett))
    return h00v


def ascat(s, Ye, teta1, teta2, he1, he2, k, dl, dl1, dl2, Ns, h0d=-1):
    H = 47.7
    Z0 = 1756
    Z1 = 8000
    tetae = max((teta1 + teta2), dl * Ye)
    teta = tetae + Ye * s
    tetal = teta1 + teta2 + Ye * s
    r1, r2 = rj(k, tetal, he1), rj(k, tetal, he2)
    if (r1 < 0.2) and (r2 < 0.2):
        return 1001
    ds = s - dl1 - dl2
    ss = (dl2 + (ds / 2)) / (dl1 + (ds / 2))
    ss = max(0.1, ss)
    z0 = ss * s * tetal / ((1 + ss) ** 2)
    ns = (z0 / Z0) * (1 + (0.031 - Ns * 2.32 * (10 ** (-3)) + (Ns ** 2) * 5.67) * np.exp(-((z0 / Z1) ** 6)))
    ett = ns
    DH0 = 6 * (0.6 - np.log10(ns) * np.log10(ss) * np.log10(r2 / (ss * r1)))
    H0 = h00(r1, r2, ett) + DH0
    if (H0 > 15) and (h0d >= 0):
        H0 = h0d
    D = s * teta

    Ascat = 10 * np.log10(k * H * (teta ** 4)) + f2(D, Ns) + H0

    return Ascat, H0


def difracton_atenuatio(s, k, teta1, teta2, dl, Ye, dls, he1, he2, dl1, dl2, Zg, Dh, hg1, hg2, Xae):
    # Constante especifica #
    tetae = max((teta1 + teta2), - dl * Ye)
    teta = tetae + s * Ye
    d3 = max(dls, dl + 1.3787 * Xae)
    d4 = d3 + 2.7574 * Xae

    # Efeito da curvatura da terra #
    if s > dl:
        Y0 = teta / (s - dl)
        Y1, Y2 = Yj(he1, dl1), Yj(he2, dl2)
        alpha0, alpha1, alpha2 = alphaj(k, Y0), alphaj(k, Y1), alphaj(k, Y2)
        K0, K1, K2 = kj(alpha0, Zg), kj(alpha1, Zg), kj(alpha2, Zg)
        x1, x2 = xj(alpha1, Y1, dl1, K1), xj(alpha2, Y2, dl2, K2)
        x0 = x(alpha0, K0, teta, x1, x2)
        Ar = g(x0) - f(x1, K1) - f(x2, K2) + c1(K0)
    else:
        Ar = 0

    # Efe ito ponta de faca #
    A3 = adif(d3, k, Dh, he1, he2, hg1, hg2, dl, tetae, Ye, dls, dl1, dl2, teta, Ar)
    A4 = adif(d4, k, Dh, he1, he2, hg1, hg2, dl, tetae, Ye, dls, dl1, dl2, teta, Ar)
    md = (A4 - A3) / (d4 - d3)
    Aed = A3 - md * d3
    Aref = Aed + md * s
    return Aref, Aed, md


def los_atenuatio(s, k, teta1, teta2, dl, Ye, dls, he1, he2, dl1, dl2, Zg, Dh, hg1, hg2, Xae):
    d2 = dls
    Arefd, Aed, md = difracton_atenuatio(s, k, teta1, teta2, dl, Ye, dls, he1, he2, dl1, dl2, Zg, Dh, hg1, hg2, Xae)
    A2 = Aed + md * d2
    AK1 = 0
    AK2 = 0
    if Aed >= 0:
        d0 = min(0.5 * dl, 1.908 * k * he1 * he2)
        d1 = (3 / 4) * d0 + (1 / 4) * dl
        A0 = alos(k, dls, Dh, d0, md, Aed, he1, he2, Zg)
        A1 = alos(k, dls, Dh, d1, md, Aed, he1, he2, Zg)

        K2l = max(0, ((d2 - d0) * (A1 - A0) - (d1 - d0) * (A2 - A0)) / (
                (d2 - d0) * np.log(d1 / d0) - (d1 - d0) * np.log(d2 / d0)))
        K1l = (A2 - A0 - K2l * np.log(d2 / d0)) / (d2 - d0)

        if K1l >= 0:
            AK1 = K1l
            AK2 = K2l
        else:
            K2ll = (A2 - A0) / np.log(d2 / d0)
            if K2ll >= 0:
                AK1 = 0
                AK2 = K2ll
            else:
                AK1 = md
                AK2 = 0
    else:
        d0 = 1.908 * k * he1 * he2
        d1 = max(-Aed / md, dl / 4)
        if d0 < d1:
            A0 = alos(k, dls, Dh, d0, md, Aed, he1, he2, Zg)
            A1 = alos(k, dls, Dh, d1, md, Aed, he1, he2, Zg)
            K2l = max(0, ((d2 - d0) * (A1 - A0) - (d1 - d0) * (A2 - A0)) / (d2 - d0) * np.log(d1 / d0) - (
                    d1 - d0) * np.log(d2 / d0))
            if K2l > 0:
                K1l = (A2 - A0 - K2l * np.log(d2 / d0)) / (d2 - d0)
                if K1l >= 0:
                    AK1 = K1l
                    AK2 = K2l
                else:
                    K2ll = (A2 - A0) / np.log(d2 / d0)
                    if K2ll >= 0:
                        AK1 = 0
                        AK2 = K2ll
                    else:
                        AK1 = md
                        AK2 = 0

            else:
                K1ll = (A2 - A1) / (d2 - d1)
                if K1ll > 0:
                    AK1 = K1ll
                    AK2 = 0
                else:
                    AK1 = md
                    AK2 = 0
        else:
            A1 = alos(k, dls, Dh, d1, md, Aed, he1, he2, Zg)
            K1ll = (A2 - A1) / (d2 - d1)
            if K1ll > 0:
                AK1 = K1ll
                AK2 = 0
            else:
                AK1 = md
                AK2 = 0
    Ael = A2 - AK1 * d2

    Aref = max(0, Ael + AK1 * s + AK2 * np.log(s / dls))
    return Aref


def scatter_atenuatio(s, k, teta1, teta2, dl, Ye, dls, he1, he2, dl1, dl2, Zg, Dh, hg1, hg2, Xae, Ns):
    Hs = 47.7
    Ds = 200000
    d5 = dl + Ds
    d6 = d5 + Ds
    Arefd, Aed, md = difracton_atenuatio(s, k, teta1, teta2, dl, Ye, dls, he1, he2, dl1, dl2, Zg, Dh, hg1, hg2, Xae)

    A5, h0d5 = ascat(d5, Ye, teta1, teta2, he1, he2, k, dl, dl1, dl2, Ns)

    A6, h0d6 = ascat(d6, Ye, teta1, teta2, he1, he2, k, dl, dl1, dl2, Ns, h0d5)

    if A5 < 1000:
        ms = (A6 - A5) / Ds
        dx = max(dls, dl + Xae * np.log10(k * Hs), (A5 - Aed - ms * d5) / (md - ms))
        Aes = Aed + (md - ms) * dx
    else:
        ms = md
        Aes = Aed
        dx = 10e6

    Aref = Aes + ms * s
    return Aref, dx


def urbam_factor(d, f):  # f em MHz e d em Km
    return 16.5 + 10 * np.log10(f / 100) - 0.12 * d


def atenuaco_por_icertezas_sitacao(qs, he1, he2, k, d, yt):
    # qs de 0 a 9 sendo que 0 representa 100% e 1 a 9 representa de 10 a 90 %
    tabela_z = [-4, 1.18, 0.84, 0.52, 0.25, 0, -0.25, -0.52, -0.84, -1.18]
    D = 100000
    d0 = 130000
    a1 = 9000000
    d1 = 1266000
    vmed = 1  # para valores até 100 k  vmed é proximo de zero (AMARAL, 2012, pag 22) vamos adotar 1 por padrão

    dex = (2 * a1 * he1) ** 0.5 + (2 * a1 * he2) ** 0.5 + a1 * ((k * d1) ** (-1 / 3))

    if d <= dex:
        de = d0 * d / dex
    else:
        de = d0 + d - dex
    zs = tabela_z[qs]
    sigmas = 5 + 3 * (np.exp(-de / D))
    ys = (((sigmas ** 2) + (yt ** 2)) ** 0.5) * zs

    return ys


def longLq_rice_model(h0, f, hg1, hg2, he1, he2, d, yt, qs, dl1, dl2, Dh, visada,
                      teta1, teta2, polarizacao='v', simplificado=0):  # de 20M a 20 GHz, f em MHz
    # Constantes Gerais #
    # d em metros -> conferir
    s = d
    f0 = 47.7  # em MHz.m
    k = f / f0
    K = 4 / 3
    N0 = 320  # padaro para continenteal sub-tropical
    z1 = 9460
    N1 = 179.3
    Ns = N0*np.exp(-h0/z1)
    Ya = 157e-9  # 1/raio
    sigma = 0.005  # S/m
    er = 15
    Z0 = 376.62  # Ohms
    erlinha = er + ((Z0 * sigma / k) * 1j)
    if polarizacao == 'v':
        Zg = ((erlinha - 1) ** 0.5) / erlinha
    else:
        Zg = (erlinha - 1) ** 0.5
    Ye = Ya*(1-0.04555*np.exp(Ns/N1))  # curvatura efetiva da terra em m^-1
    dls1 = (2 * he1 / Ye) ** 0.5
    dls2 = (2 * he2 / Ye) ** 0.5
    dls = dls1 + dls2

    if visada:
        dl1 = dls1 * np.exp(-0.07 * ((Dh / max(he1, 5)) ** 0.5))
        teta1 = (0.65 * Dh * ((dls1 / dl1) - 1) - 2 * he1) / dls1
        dl2 = dls2 * np.exp(-0.07 * ((Dh / max(he2, 5)) ** 0.5))
        teta2 = (0.65 * Dh * ((dls2 / dl2) - 1) - 2 * he2) / dls2
        if d > dl1 + dl2:
            dl1 = dl1 * (d / (dl1 + dl2))
            dl2 = dl2 * (d / (dl1 + dl2))

    dl = dl1 + dl2

    Xae = (k * (Ye ** 2)) ** (-1 / 3)
    tetae = max((teta1 + teta2), -dl * Ye)
    Ascat, dx = scatter_atenuatio(s, k, teta1, teta2, dl, Ye, dls, he1, he2, dl1, dl2, Zg, Dh, hg1, hg2, Xae, Ns)

    if simplificado:
        if d <= dls:
            Aref = 20 * np.log10(1 + (dl * Dh / (he1 * he2)))
        elif (d > dls) and (d <= dx):
            a = 6370 / (1 - 0.04665 * np.exp(0.005577 * Ns))
            Aref = (1 + 0.045 * ((Dh / (c / (f*1000000))) ** 0.5) * (((a * tetae + dl) / d) ** 0.5))**(-1)
        else:
            H0 = 1 / (he1 * he2 * tetae * f * abs(0.007 - 0.058 * tetae))
            Aref = H0 + 10 * np.log10(f * (tetae ** 4)) - 0.1 * (Ns - 301) * np.exp(-tetae * d / 40)
    else:
        if d <= dls:
            Aref = los_atenuatio(s, k, teta1, teta2, dl, Ye, dls, he1, he2, dl1, dl2, Zg, Dh, hg1, hg2, Xae)
        elif (d > dls) and (d <= dx):
            Aref = difracton_atenuatio(s, k, teta1, teta2, dl, Ye, dls, he1, he2, dl1, dl2, Zg, Dh, hg1, hg2, Xae)[0]
        else:
            Aref = scatter_atenuatio(s, k, teta1, teta2, dl, Ye, dls, he1, he2, dl1, dl2, Zg, Dh, hg1, hg2, Xae, Ns)[0]

    yts = atenuaco_por_icertezas_sitacao(qs, he1, he2, k, d, yt)

    variabilidade_da_situacao = -yts

    return Aref, variabilidade_da_situacao


def ikegami_model(h, hr, f, w=22, lr=2, th=np.pi / 2):  # f em MHz
    # h-predio, hr-receptor em m e f em MHz
    # th é o angulo entre linha de visada e a rua dado em radianos pois é argumento de np.sin()
    # w é a largura da rua
    l = - 5.8 - 10 * np.log10(1 + (3 / (lr ** 2))) - 10 * np.log10(w) + 20 * np.log10(h - hr) + 10 * np.log10(
        np.sin(th)) + 10 * np.log10(f)
    return l

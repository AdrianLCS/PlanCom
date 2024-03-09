import numpy as np

c = 299792458  # m/s


def raio_fresnel(n, d1, d2, f):
    # f em Mhertz
    return (n * (c / (f*1000000)) * d1 * d2 / (d1 + d2)) ** 0.5


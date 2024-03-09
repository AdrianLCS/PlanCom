import numpy as np

c = 299792458  # m/s


def raio_fresnel(n, d1, d2, f):
    # f em hertz
    return (n * (c / f) * d1 * d2 / (d1 + d2)) ** 0.5


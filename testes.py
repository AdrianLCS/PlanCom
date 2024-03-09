import ee
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Autentique a conta do Earth Engine

ee.Initialize(project="plancom-409417")


def reta(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    v = p2 - p1
    modulo = np.linalg.norm(v)
    n = int(np.ceil(modulo / 0.00027776))  # precisao de 30 m
    t = np.linspace(0, 1, n)
    r = []
    for i in t:
        r.append(p1 + v * i)
    r = np.array(r)
    return r


def perfil(r, src):
    raster = src.read(1)
    inv_transform = ~src.transform
    dem1 = []
    dem2 = []
    dem3 = []
    dsm = []
    landcover = []
    d = []
    ponto0 = ee.Geometry.Point(r[0][0], r[0][1])
    # elv = ee.Image('USGS/SRTMGL1_003')
    elv2 = ee.Image("NASA/NASADEM_HGT/001")
    # altura = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM')
    # lcover10 = ee.ImageCollection("ESA/WorldCover/v200")
    for i in range(np.shape(r)[0]):
        pixel_x1, pixel_y1 = inv_transform * (r[i][0], r[i][1])
        ponto = ee.Geometry.Point(r[i][0], r[i][1])
        d.append(ponto0.distance(ponto).getInfo())
        # dem1.append(elv.sample(ponto, 30).first().get('elevation').getInfo())
        dem2.append(elv2.sample(ponto, 30).first().get('elevation').getInfo())
        #dem3.append(raster[int(np.floor(pixel_y1))][int(np.ceil(pixel_x1))])
        # dsm.append(altura.mean().sample(ponto, 30).first().get('DSM').getInfo())
        """landcover.append(lcover10.first().sample(ponto, 10).first().get('Map').getInfo())
        if i < np.shape(r)[0] - 1:
            lonpasso = (r[i + 1][0] - r[i][0]) / 3
            latpasso = (r[i + 1][1] - r[i][1]) / 3
            ponto2 = ee.Geometry.Point(r[i][0] + lonpasso, r[i][1] + latpasso)
            ponto3 = ee.Geometry.Point(r[i][0] + 2 * lonpasso, r[i][1] + 2 * latpasso)
            landcover.append(lcover10.first().sample(ponto2, 10).first().get('Map').getInfo())
            landcover.append(lcover10.first().sample(ponto3, 10).first().get('Map').getInfo())"""

    return dem1, dem2, dem3, dsm, landcover, d


def ajuste(elevacao, distancia, hg1, hg2, dl1, dl2):
    xa = int(min(15 * hg1, 0.1 * dl1) / distancia[1])
    xb = len(elevacao) - 1 - int(min(15 * hg2, 0.1 * dl2) / distancia[1])
    zorig = elevacao[xa:xb + 1]
    xorig = np.array(range(xa, xb + 1))
    z = []
    x = []
    u = 0
    while u < len(xorig):
        xaux = np.mean(xorig[u])
        zaux = np.mean(zorig[u])
        x.append(xaux)
        z.append(zaux)
        u = u + 1
    z = np.array(z)
    x = np.array(x)
    # ajuste

    n = len(x)
    sx = 0
    sy = 0
    syx = 0
    s2x = 0
    for i in range(len(x)):
        sx = sx + x[i]
        sy = sy + z[i]
        s2x = s2x + x[i] ** 2
        syx = syx + z[i] * x[i]

    c1 = ((syx / (n * s2x - (sx ** 2))) * n) - ((sy / (n * s2x - (sx ** 2))) * sx)
    c0 = ((s2x / (n * s2x - (sx ** 2))) * sy) - ((syx / (n * s2x - (sx ** 2))) * sx)
    zl = c1 * x + c0

    return x, zl, z


r = reta((-43.1895, -22.9036), (-43.1661, -22.9555))
with rasterio.open('C:\PythonFlask\PlanCom\Raster\S23W044.tif') as raster:
    dem1, dem2, dem3, dsm, landcover, d = perfil(r, raster)

x, zl, z = ajuste(dem2, d, 22.5, 2, 1679.3137215642932, 183.1989813982053)

print(dem2[-1])
print(zl[-1])
# Criar uma grade de subplots (2 linhas, 2 colunas)
plt.subplot(2, 2, 1)
plt.plot(x, zl)
plt.title('ajuste')

plt.subplot(2, 2, 2)
plt.plot(d, dem2)
plt.title('DEM raster')

# Ajustar o layout para evitar sobreposição
plt.tight_layout()

# Mostrar a figura
plt.show()

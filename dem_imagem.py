import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Substitua 'seu_arquivo_raster.tif' pelo caminho do seu arquivo raster
# file_path = "C:\PythonFlask\PlanCom\Raster\S27\ASTGTMV003_S27W049_dem.tif" #ASTGTMV003_S23W044_dem long de 22 a 23 e lat de 43 a 44
file_path = "C:\PythonFlask\PlanCom\Raster\S23W044.tif"

correcao= 6228.6112900782355/5693.127435777792
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
    transform = src.transform
    inv_transform = ~src.transform
    unidade_distancia=transform[0]* (10 ** 5)*correcao
    x0, y0 = inv_transform * (r[0][0], r[0][1])
    dem_raster = []
    d = []
    for i in range(np.shape(r)[0]):
        pixel_x1, pixel_y1 = inv_transform * (r[i][0], r[i][1])
        dist=unidade_distancia  * ((((pixel_x1-x0) ** 2) + ((pixel_y1-y0) ** 2)) ** 0.5)
        d.append(dist)
        alt = ((1-(pixel_y1 - np.floor(pixel_y1))) * raster[int(np.floor(pixel_y1))][int(np.floor(pixel_x1))] + (
            pixel_y1 - np.floor(pixel_y1)) * raster[int(np.ceil(pixel_y1))][int(np.floor(pixel_x1))] + (
                1-(pixel_x1 - np.floor(pixel_x1))) * raster[int(np.floor(pixel_y1))][int(np.floor(pixel_x1))] + (
                          pixel_x1 - np.floor(pixel_x1)) * raster[int(np.floor(pixel_y1))][int(np.ceil(pixel_x1))])/2

        dem_raster.append(alt)
    return d, dem_raster


def obter_perfil(ponto_inicial, ponto_final, src):
    raster = src.read(1)
    transform = src.transform
    inv_transform = ~src.transform
    pixel_x1, pixel_y1 = inv_transform * (ponto_inicial[0], ponto_inicial[1])
    pixel_x2, pixel_y2 = inv_transform * (ponto_final[0], ponto_final[1])
    pixel_x1, pixel_y1 = np.floor(pixel_x1), np.floor(pixel_y1)
    pixel_x2, pixel_y2 = np.floor(pixel_x2), np.floor(pixel_y2)

    if pixel_x1 <= pixel_x2:
        vet_x = np.arange(pixel_x1, pixel_x2 + 1, 1)
    else:
        vet_x = np.arange(pixel_x1, pixel_x2 - 1, - 1)
    if pixel_y1 <= pixel_y2:
        vet_y = np.arange(pixel_y1, pixel_y2 + 1, 1)
    else:
        vet_y = np.arange(pixel_y1, pixel_y2 - 1, -1)
    tamx = len(vet_x)
    tamy = len(vet_y)
    razao = max(tamx, tamy) / min(tamx, tamy)
    # Converte a distância de pixels para a unidade do raster
    unidade_distancia = transform[0]  # A primeira componente da transformação representa a resolução do pixel

    if tamx >= tamy:
        id = 'x'
        unidade_distancia = unidade_distancia * (10 ** 5) * (((tamx ** 2) + (tamy ** 2)) ** 0.5) / tamx
        distancia = np.array(range(tamx))
        distancia = distancia * unidade_distancia
        elevacao = np.array(range(tamx))
        xcoordg = np.array(range(tamx))
        ycoordg = np.array(range(tamx))
    else:
        unidade_distancia = unidade_distancia * (10 ** 5) * (((tamx ** 2) + (tamy ** 2)) ** 0.5) / tamy
        elevacao = np.array(range(tamy))
        id = 'y'
        distancia = np.array(range(tamy))
        distancia = distancia * unidade_distancia
        xcoordg = np.array(range(tamx))
        ycoordg = np.array(range(tamx))

    # xcoord = np.array(vet_x)
    # ycoord = np.array(vet_y)
    contmin = 0
    contmax = 0
    if id == 'x':
        for i in elevacao:
            indice = (contmin + 1) * razao - (contmax + 1)
            # xcoord[contmax], ycoord[contmin] = transform * (vet_x[contmax], vet_y[contmin])

            if indice > 0:
                elevacao[i] = raster[int(vet_y[contmin])][int(vet_x[contmax])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmax], vet_y[contmin])
            elif indice == 0:
                elevacao[i] = raster[int(vet_y[contmin])][int(vet_x[contmax])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmax], vet_y[contmin])
                contmin = contmin + 1
            else:
                elevacao[i] = (1 + indice) * raster[int(vet_y[contmin])][int(vet_x[contmax])] - indice * \
                              raster[int(vet_y[contmin + 1])][int(vet_x[contmax])]
                xcoordg[i], ycoordg[i] = (1 + indice) * transform * (
                    vet_x[contmax], vet_y[contmin]) - indice * transform * (vet_x[contmax], vet_y[contmin + 1])
                contmin = contmin + 1
            contmax = contmax + 1
    else:
        for i in elevacao:
            indice = (contmin + 1) * razao - (contmax + 1)
            # xcoord[contmin], ycoord[contmax] = transform * (vet_x[contmin], vet_y[contmax])
            if indice > 0:
                elevacao[i] = raster[int(vet_y[contmax])][int(vet_x[contmin])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmin], vet_y[contmax])
            elif indice == 0:
                elevacao[i] = raster[int(vet_y[contmax])][int(vet_x[contmin])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmin], vet_y[contmax])
                contmin = contmin + 1
            else:
                elevacao[i] = (1 + indice) * raster[int(vet_y[contmax])][int(vet_x[contmin])] - indice * \
                              raster[int(vet_y[contmax])][int(vet_x[contmin + 1])]
                xcoordg[i], ycoordg[i] = (1 + indice) * transform * (
                    vet_x[contmin], vet_y[contmax]) - indice * transform * (vet_x[contmin + 1], vet_y[contmax])
                contmin = contmin + 1

            contmax = contmax + 1
    return distancia, elevacao, xcoordg, ycoordg


def perfil_da_elevacao(ponto_inicial, ponto_final, src):
    raster = src.read(1)
    transform = src.transform

    inv_transform = ~src.transform
    pixel_x1, pixel_y1 = inv_transform * (ponto_inicial[0], ponto_inicial[1])
    pixel_x2, pixel_y2 = inv_transform * (ponto_final[0], ponto_final[1])
    pixel_x1, pixel_y1 = np.floor(pixel_x1), np.floor(pixel_y1)
    pixel_x2, pixel_y2 = np.floor(pixel_x2), np.floor(pixel_y2)

    # Calcula a distância euclidiana entre os pixels
    distancia_pixel = ((pixel_x2 - pixel_x1) ** 2 + (pixel_y2 - pixel_y1) ** 2) ** 0.5

    print(((82 ** 2) + 1) ** 0.5)
    if pixel_x1 <= pixel_x2:
        vet_x = np.arange(pixel_x1, pixel_x2 + 1, 1)
    else:
        vet_x = np.arange(pixel_x1, pixel_x2 - 1, - 1)
    if pixel_y1 <= pixel_y2:
        vet_y = np.arange(pixel_y1, pixel_y2 + 1, 1)
    else:
        vet_y = np.arange(pixel_y1, pixel_y2 - 1, -1)
    tamx = len(vet_x)
    tamy = len(vet_y)
    razao = max(tamx, tamy) / min(tamx, tamy)
    elevacao = np.array(range(max(tamx, tamy)))
    # Converte a distância de pixels para a unidade do raster
    unidade_distancia = transform[0]  # A primeira componente da transformação representa a resolução do pixel

    if tamx >= tamy:
        id = 'x'
        unidade_distancia = unidade_distancia * (10 ** 5) * (((tamx ** 2) + (tamy ** 2)) ** 0.5) / tamx
        distancia = np.arange(0.0, tamy)
        distancia = distancia * unidade_distancia
        elevacao = np.array(range(tamx))
        xcoordg = np.arange(0.0, tamy)
        ycoordg = np.arange(0.0, tamy)
    else:
        unidade_distancia = unidade_distancia * (10 ** 5) * (((tamx ** 2) + (tamy ** 2)) ** 0.5) / tamy
        elevacao = np.arange(0.0, tamy)
        id = 'y'
        distancia = np.arange(0.0, tamy)
        distancia = distancia * unidade_distancia
        xcoordg = np.arange(0.0, tamy)
        ycoordg = np.arange(0.0, tamy)

    xcoord = np.array(vet_x)
    ycoord = np.array(vet_y)
    contmin = 0
    contmax = 0
    if id == 'x':
        for i in elevacao:
            indice = (contmin + 1) * razao - (contmax + 1)
            xcoord[contmax], ycoord[contmin] = transform * (vet_x[contmax], vet_y[contmin])
            if indice > 0:
                elevacao[i] = raster[int(vet_y[contmin])][int(vet_x[contmax])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmax], vet_y[contmin])
            elif indice == 0:
                elevacao[i] = raster[int(vet_y[contmin])][int(vet_x[contmax])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmax], vet_y[contmin])
                contmin = contmin + 1
            else:
                # elevacao[i] = (1+indice)*raster[int(vet_y[contmin])][int(vet_x[contmax])] - indice*raster[int(vet_y[contmin+1])][int(vet_x[contmax])]
                elevacao[i] = raster[int(vet_y[contmin + 1])][int(vet_x[contmax])]

                xcoordg[i], ycoordg[i] = transform * (vet_x[contmax], vet_y[contmin + 1])
                contmin = contmin + 1
            contmax = contmax + 1
    else:
        for i in elevacao:
            i = int(i)
            indice = (contmin + 1) * razao - (contmax + 1)
            xcoord[contmin], ycoord[contmax] = transform * (vet_x[contmin], vet_y[contmax])
            if indice > 0:
                elevacao[i] = raster[int(vet_y[contmax])][int(vet_x[contmin])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmin], vet_y[contmax])
            elif indice == 0:
                elevacao[i] = raster[int(vet_y[contmax])][int(vet_x[contmin])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmin], vet_y[contmax])
                contmin = contmin + 1
            else:
                # elevacao[i] = (1+indice)*raster[int(vet_y[contmax])][int(vet_x[contmin])] - indice*raster[int(vet_y[contmax])][int(vet_x[contmin+1])]
                elevacao[i] = raster[int(vet_y[contmax])][int(vet_x[contmin + 1])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmin + 1], vet_y[contmax])
                contmin = contmin + 1

            contmax = contmax + 1

    return distancia, elevacao


ponto_inicial = (-43.972385, -22.055411)  # -22.055411, -43.972385
ponto_final = (-43.938665, -22.990583)  # -22.736391, -43.515914
# Abrir o arquivo raster
with rasterio.open(file_path) as src:
    # Ler os dados raster como uma matriz numpy
    raster_data = src.read(1)
    print(np.shape(raster_data))
    transform = src.transform
    r=reta(ponto_inicial, ponto_final)
    dist, elev = perfil(r, src)
    #dist, elev = perfil_da_elevacao(ponto_inicial, ponto_final, src)

    ###OBTER COORDENADAS TENDO OS PIXELS
    # Especifica as coordenadas do pixel desejado
    pixel_x, pixel_y = (2918.3000000000175, 3253.460000000021)  # 1743, 2651
    # Calcula as coordenadas espaciais correspondentes
    x, y = transform * (pixel_x, pixel_y)
    print(f'Coordenadas (x, y) do pixel ({pixel_x}, {pixel_y}): ({x}, {y})')

    ###OBTER O PIXEL TENDO AS COORDENADAS
    inv_transform = ~src.transform
    # Especifica a latitude e a longitude desejadas
    #latitude, longitude = (-22.9036, -43.1895)
    # Calcula as coordenadas do pixel correspondentes
    #pixel_x, pixel_y = inv_transform * (longitude, latitude)
    #print(f'Pixel correspondente à latitude {latitude} e longitude {longitude}: ({pixel_x}, {pixel_y})')
    print(raster_data[int(pixel_y)][int(pixel_x)])
    lon2, lat2 = src.xy(ponto_final[0], ponto_final[1])
    # Obter informações sobre a matriz raster (extensão, resolução, etc.)
    print(src.profile)
# ober altura
print(dist[-1])
# Visualizar o raster usando Matplotlib
plt.imshow(raster_data, cmap='terrain')
plt.colorbar(label='Elevação (metros)')
#plt.plot(dist, elev)
plt.title('Modelo Digital de Elevação (DEM)')

plt.show()

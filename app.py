import numpy as np
import rasterio
from flask import Flask, render_template, request, jsonify
import os
import ee
import folium
from owslib.wms import WebMapService
import itmModel
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from folium import plugins
#correcao distancia 6228.6112900782355/5712.1356899878 ou 5719.711764799506

# no marcador definir o radio. radio sera um objeto que contém potencia ganho da antena limear de recepcao

# predições somento na américa do Sul
"""
#### DISTANCIA COM EE####
ponto_inicial = ee.Geometry.Point(-22.05541666666666, -43.97236111111112)
ponto_final = ee.Geometry.Point(-22.7364, -43.5159)
distancia_planar = ponto_inicial.distance(ponto_final).getInfo()
print(distancia_planar)
"""

ee.Initialize(project="plancom-409417")
app = Flask(__name__)
c = 299792458  # m/s

from rasterio.transform import from_origin


def calcula_vetores_area(yxp1, yxp2, matriz_dem, matriz_dsm, matriz_dist, matriz_landcover):
    tamx = int(abs(yxp2[1] - yxp1[1]) + 1)
    if yxp2[1] - yxp1[1] >= 0:
        sinalx = int(1)
    else:
        sinalx = int(-1)
    tamy = int(abs(yxp2[0] - yxp1[0]) + 1)
    if yxp2[0] - yxp1[0] >= 0:
        sinaly = int(1)
    else:
        sinaly = int(-1)
    razao = max(tamx, tamy) / min(tamx, tamy)
    contmin = 0
    contmax = 0
    iy = int(yxp1[0])
    ix = int(yxp1[1])
    if tamx >= tamy:
        distancia = np.arange(0.0, tamx)
        elevacao = np.arange(0.0, tamx)
        dsm = np.arange(0.0, tamx)
        landcover = np.arange(0.0, tamx)
        for i in elevacao:
            i = int(i)
            indice = (contmin + 1) * razao - (contmax + 1)
            # xcoord[contmax], ycoord[contmin] = transform * (vet_x[contmax], vet_y[contmin])
            if indice > 0:
                elevacao[i] = matriz_dem[iy][ix]
                distancia[i] = matriz_dist[iy][ix]
                dsm[i] = matriz_dsm[iy][ix]
                landcover[i] = matriz_landcover[iy][ix]
            elif indice == 0:
                elevacao[i] = matriz_dem[iy][ix]
                distancia[i] = matriz_dist[iy][ix]
                dsm[i] = matriz_dsm[iy][ix]
                landcover[i] = matriz_landcover[iy][ix]
                iy = int(iy + sinaly)
                contmin = contmin + 1
            else:
                iy = int(iy + sinaly)
                contmin = contmin + 1
                elevacao[i] = matriz_dem[iy][ix]
                distancia[i] = matriz_dist[iy][ix]
                dsm[i] = matriz_dsm[iy][ix]
                landcover[i] = matriz_landcover[iy][ix]
            ix = int(ix + sinalx)
            contmax = contmax + 1
    else:
        distancia = np.arange(0.0, tamy)
        elevacao = np.arange(0.0, tamy)
        dsm = np.arange(0.0, tamy)
        landcover = np.arange(0.0, tamy)
        for i in elevacao:
            i = int(i)
            indice = (contmin + 1) * razao - (contmax + 1)
            # xcoord[contmin], ycoord[contmax] = transform * (vet_x[contmin], vet_y[contmax])
            if indice > 0:
                elevacao[i] = matriz_dem[iy][ix]
                distancia[i] = matriz_dist[iy][ix]
                dsm[i] = matriz_dsm[iy][ix]
                landcover[i] = matriz_landcover[iy][ix]
            elif indice == 0:
                elevacao[i] = matriz_dem[iy][ix]
                distancia[i] = matriz_dist[iy][ix]
                dsm[i] = matriz_dsm[iy][ix]
                landcover[i] = matriz_landcover[iy][ix]
                contmin = contmin + 1
                ix = int(ix + sinalx)
            else:
                ix = int(ix + sinalx)
                contmin = contmin + 1
                elevacao[i] = matriz_dem[iy][ix]
                distancia[i] = matriz_dist[iy][ix]
                dsm[i] = matriz_dsm[iy][ix]
                landcover[i] = matriz_landcover[iy][ix]
            iy = int(iy + sinaly)
            contmax = contmax + 1

    return elevacao, dsm, landcover, distancia


def calcula_perda(ht, hr, f, yxp1, yxp2, matriz_dem, matriz_dsm, matriz_dist, matriz_landcover):
    dem, dsm, landcover, distancia = calcula_vetores_area(yxp1, yxp2, matriz_dem, matriz_dsm, matriz_dist,
                                                          matriz_landcover)
    d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h, visada, indice_visada_r = obter_dados_do_perfil(
        dem, dsm, distancia, ht, hr)

    if landcover[-1] == 50:
        urban = 'wi'
    else:
        urban = 'n'

    yt = 1  # é a perda pelo clima, adotar esse valor padrao inicialmente
    qs = 7  # 70% das situacões
    espesura, h, d_urb, hb_urb = get_dados_landcover(indice_visada_r, dem, landcover, dsm, hr, ht, distancia, h,
                                                     dl2, visada, area=1)
    # colocar a cidicao para chamar itm ou urbano + espaco livre

    print(
        f' ({f}, {hg1}, {hg2}, {he1}, {he2}, {d}, {yt}, {qs}, {dl1}, {dl2}, {Dh}, {visada}, {h},{teta1}, {teta2}, {d_urb}, {hb_urb}, {urban})')

    perda = itmModel.longLq_rice_model(f, hg1, hg2, he1, he2, d, yt, qs, dl1, dl2, Dh, visada, h,
                                       teta1, teta2, d_urb, hb_urb, urban, polarizacao='v')

    return perda


def modificar_e_salvar_raster(raster_path, ponto, raio, limear, ht, f):
    # dem = ee.Image("NASA/NASADEM_HGT/001")
    # dsm = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM')
    # lcover10 = ee.ImageCollection("ESA/WorldCover/v200")

    pasta = raster_path[:-11] + 'modificado'
    file = '\A' + raster_path[-11:]
    # Abrir o arquivo raster para leitura e escrita
    with rasterio.open(raster_path, 'r+') as src:
        # Ler a matriz de dados do raster
        data = src.read(1)
        inv_transform = ~src.transform
        transform = src.transform
        x, y = inv_transform * (ponto[0], ponto[1])
        unidade_distancia = transform[0] * (10 ** 5)
        raio = raio / unidade_distancia
        # ponto0 = ee.Geometry.Point(ponto[0], ponto[1])
        esquerda = int(x - raio)
        direita = int(x + raio)
        cima = int(y - raio)
        baixo = int(y + raio)
        yxp1 = (int(raio), int(raio))
        matriz_dem = []
        matriz_dsm = []
        matriz_dist = []
        matriz_landcover = []
        for linha in range(cima, baixo):
            vet_dem = []
            vet_dsm = []
            vet_dist = []
            vet_landcover = []
            for coluna in range(esquerda, direita):
                lon, lat = transform * (coluna, linha)
                # ponto1 = ee.Geometry.Point(lon, lat)
                # vet_dem.append(dem.sample(ponto1, 30).first().get('elevation').getInfo())  # vet_dem.append(data[linha][coluna])
                vet_dem.append(data[linha][coluna])  # vet_dem.append(data[linha][coluna])
                # vet_dsm.append(dsm.mean().sample(ponto0, 30).first().get('DSM').getInfo())
                vet_dsm.append(data[linha][coluna])
                # vet_dist.append(ponto0.distance(ponto1).getInfo())
                vet_dist.append(
                    ((((linha - y) * unidade_distancia) ** 2) + (((coluna - x) * unidade_distancia) ** 2)) ** 0.5)
                # vet_landcover.append(lcover10.first().sample(ponto1, 30).first().get('Map').getInfo())
                vet_landcover.append(data[linha][coluna])
            matriz_dem.append(vet_dem)
            matriz_dsm.append(vet_dsm)
            matriz_dist.append(vet_dist)
            matriz_landcover.append(vet_landcover)

        # Modificar o valor do ponto desejado
        for linha in range(np.shape(data)[0]):
            for coluna in range(np.shape(data)[1]):
                if (((((linha - y) ** 2) + ((coluna - x) ** 2)) ** 0.5) < (raio - 2)) and (
                        ((((linha - y) ** 2) + ((coluna - x) ** 2)) ** 0.5) * unidade_distancia > 200):

                    p = calcula_perda(ht, 2, f, yxp1, (linha - cima, coluna - esquerda), matriz_dem,
                                      matriz_dsm, matriz_dist, matriz_landcover)
                    if p >= limear:
                        data[linha][coluna] = 100
                    else:
                        data[linha][coluna] = 10
                else:
                    data[linha][coluna] = 100

        # Atualizar os metadados do raster
        # src.write(data, 1)

        # Obter os metadados do raster original
        meta = src.meta.copy()

    # Salvar o novo raster modificado com um nome diferente
    with rasterio.open(pasta + file, 'w', **meta) as dst:
        # Copiar os dados do raster original para o novo arquivo
        with rasterio.Env():
            dst.write(data, 1)
    return pasta + file


def criaimg(dem_file):
    # Carregar o arquivo DEM (tif)
    dem_dataset = rasterio.open(dem_file)

    # Ler os dados do DEM como uma matriz NumPy
    dem_data = dem_dataset.read(1, masked=True)  # Assumindo que os dados estão no primeiro canal

    # Configurar a figura e os eixos para o gráfico sem eixos e títulos
    fig, ax = plt.subplots(figsize=(100, 100))
    ax.axis('off')

    # Criar um mapa de cores para representar as altitudes do terreno
    cmap = plt.get_cmap('terrain')
    im = ax.imshow(dem_data, cmap=cmap, extent=[dem_dataset.bounds.left, dem_dataset.bounds.right,
                                                dem_dataset.bounds.bottom, dem_dataset.bounds.top])

    # Salvar a imagem em um arquivo sem títulos, eixos e barra de cores
    plt.savefig(dem_file[:-3] + "png", format="png", bbox_inches='tight', pad_inches=0)

    # Fechar a figura para liberar recursos
    plt.close()
    return dem_file[:-3] + "png"


def criamapa(dem_file, img_file):
    # Carregar o arquivo DEM (tif)
    dem_dataset = rasterio.open(dem_file)

    # Obter as informações sobre a extensão do DEM
    bounds = dem_dataset.bounds
    min_lat, min_lon = bounds.bottom, bounds.left
    max_lat, max_lon = bounds.top, bounds.right

    # Calcular o centro do DEM para definir o local inicial do mapa
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Criar um mapa OpenStreetMap usando Folium
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')
    nome = '0'
    for i in cobertura:
        if i['img'] == img_file:
            nome = i['nome']

    # Adicionar a camada do raster como uma sobreposição de imagem
    image_overlay = folium.raster_layers.ImageOverlay(
        image=img_file,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.5,
        name=nome,
        interactive=True,
        cross_origin=False,
        zindex=1
    )

    return image_overlay


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
    dem_raster = []
    dem = []
    dsm = []
    landcover = []
    d = []
    ponto0 = ee.Geometry.Point(r[0][0], r[0][1])
    elv2 = ee.Image("NASA/NASADEM_HGT/001")
    altura = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM')
    lcover10 = ee.ImageCollection("ESA/WorldCover/v200")
    print(ponto0.distance(ee.Geometry.Point(r[-1][0],r[-1][1])).getInfo())
    for i in range(np.shape(r)[0]):
        ponto = ee.Geometry.Point(r[i][0], r[i][1])
        pixel_x1, pixel_y1 = inv_transform * (r[i][0], r[i][1])
        d.append(ponto0.distance(ponto).getInfo())
        dem.append(elv2.sample(ponto, 30).first().get('elevation').getInfo())
        dsm.append(altura.mean().sample(ponto, 30).first().get('DSM').getInfo())
        dem_raster.append(raster[int(pixel_y1)][int(pixel_x1)])
        landcover.append(lcover10.first().sample(ponto, 10).first().get('Map').getInfo())
        if i < np.shape(r)[0] - 1:
            lonpasso = (r[i + 1][0] - r[i][0]) / 3
            latpasso = (r[i + 1][1] - r[i][1]) / 3
            ponto2 = ee.Geometry.Point(r[i][0] + lonpasso, r[i][1] + latpasso)
            ponto3 = ee.Geometry.Point(r[i][0] + 2 * lonpasso, r[i][1] + 2 * latpasso)
            landcover.append(lcover10.first().sample(ponto2, 10).first().get('Map').getInfo())
            landcover.append(lcover10.first().sample(ponto3, 10).first().get('Map').getInfo())

    return dem, dsm, landcover, d, dem_raster


def raio_fresnel(n, d1, d2, f):
    # f em hertz
    return (n * (c / f) * d1 * d2 / (d1 + d2)) ** 0.5


def unir_raster2x2(file1, file2, file3, file4):
    # Nome do arquivo de saída
    output_raster = file1 + '_unido' + '.tif'
    # Abrir os quatro rasters
    with rasterio.open(file1) as src1, rasterio.open(file2) as src2, \
            rasterio.open(file3) as src3, rasterio.open(file4) as src4:
        # Obter informações sobre um dos rasters (usaremos raster1 como referência)
        profile = src1.profile
        transform = src1.transform
        # Obter os dados raster de cada arquivo
        data1 = src1.read(1)
        data2 = src2.read(1)
        data3 = src3.read(1)
        data4 = src4.read(1)
        # print(data4.shape)
        # Criar um array 2D para armazenar os dados combinados
        # combined_data = np.zeros_like(data1)
        combined_data = np.zeros((7202, 7202))
        # Inserir os dados de cada raster na posição apropriada no array combinado
        combined_data[:src1.height, :src1.width] = data1
        combined_data[:src2.height, src1.width:] = data2
        combined_data[src1.height:, :src3.width] = data3
        combined_data[src1.height:, src3.width:] = data4
    # Atualizar as informações do perfil do raster
    profile.update(count=1, width=combined_data.shape[1], height=combined_data.shape[0], transform=transform)
    # Salvar o raster combinado
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(combined_data, 1)
    return output_raster


def aterar_caracter(minha_string, indice_alteracao, novo_caractere):
    return minha_string[:indice_alteracao] + novo_caractere + minha_string[indice_alteracao + 1:]


def mudar_ns(raster, mm):
    if raster[2] == 0 and raster[2] + mm == -1:
        raster = aterar_caracter(raster, 2, str(abs(int(raster[2]) + mm)))
        raster = aterar_caracter(raster, 0, 'S')
        return raster
    elif raster[2] == 1 and raster[2] + mm == 0 and raster[0] == 'S':
        raster = aterar_caracter(raster, 2, str(abs(int(raster[2]) + mm)))
        raster = aterar_caracter(raster, 0, 'N')
        return raster
    else:
        return raster[2] + mm


def unir_raster(raster1, raster2, ref):
    file1, file2, file3, file4 = '', '', '', ''
    if abs(int(raster1[2]) - int(raster2[2])) <= 1 and abs(int(raster1[6]) - int(raster2[6])) <= 1:
        # unir raster 2x2, files:
        # 1  2
        # 3  4
        if raster1[0] == 'S' and raster1[3] == 'W':
            if ref == ('b' or 'd' or 'db'):
                file1 = raster1
                file2 = aterar_caracter(raster1, 6, str(int(raster1[6]) - 1))
                file3 = aterar_caracter(raster1, 2, str(int(raster1[2]) + 1))
                file4 = aterar_caracter(file2, 2, str(int(file2[2]) + 1))
            elif ref == ('c' or 'e' or 'ec'):
                file4 = raster1
                file2 = mudar_ns(raster1, -1)
                file3 = aterar_caracter(raster1, 6, str(int(raster1[6]) + 1))
                file1 = aterar_caracter(file2, 6, str(int(file2[6]) + 1))
            elif ref == 'dc':
                file3 = raster1
                file1 = mudar_ns(raster1, -1)
                file4 = aterar_caracter(raster1, 6, str(int(raster1[6]) - 1))
                file2 = aterar_caracter(file1, 6, str(int(file1[6]) - 1))
            elif ref == 'eb':
                file2 = raster1
                file4 = aterar_caracter(raster1, 2, str(int(raster1[2]) + 1))
                file1 = aterar_caracter(raster1, 6, str(int(raster1[6]) + 1))
                file3 = aterar_caracter(file4, 6, str(int(file4[6]) + 1))
        elif raster1[0] == 'N' and raster1[3] == 'W':
            if ref == ('b' or 'd' or 'db'):
                file1 = raster1
                file2 = aterar_caracter(raster1, 6, str(int(raster1[6]) - 1))
                file3 = mudar_ns(raster1, -1)
                file4 = aterar_caracter(file2, 2, str(int(file2[2]) + 1))
            elif ref == ('c' or 'e' or 'ec'):
                file4 = raster1
                file2 = aterar_caracter(raster1, 2, str(int(raster1[2]) + 1))
                file3 = aterar_caracter(raster1, 6, str(int(raster1[6]) + 1))
                file1 = aterar_caracter(file2, 6, str(int(file2[6]) + 1))
            elif ref == 'dc':
                file3 = raster1
                file1 = aterar_caracter(raster1, 2, str(int(raster1[2]) + 1))
                file4 = aterar_caracter(raster1, 6, str(int(raster1[6]) - 1))
                file2 = aterar_caracter(file1, 6, str(int(file1[6]) - 1))
            elif ref == 'eb':
                file2 = raster1
                file4 = mudar_ns(raster1, -1)  # se esse int der negativo colocar abs e trocae n pelo s
                file1 = aterar_caracter(raster1, 6, str(int(raster1[6]) + 1))
                file3 = aterar_caracter(file4, 6, str(int(file4[6]) + 1))

    file1 = os.path.join('C:\PythonFlask\PlanCom\Raster', file1)
    file2 = os.path.join('C:\PythonFlask\PlanCom\Raster', file2)
    file3 = os.path.join('C:\PythonFlask\PlanCom\Raster', file3)
    file4 = os.path.join('C:\PythonFlask\PlanCom\Raster', file4)
    file1dsm = os.path.join('C:\PythonFlask\PlanCom\dsm', file1)
    file2dsm = os.path.join('C:\PythonFlask\PlanCom\dsm', file2)
    file3dsm = os.path.join('C:\PythonFlask\PlanCom\dsm', file3)
    file4dsm = os.path.join('C:\PythonFlask\PlanCom\dsm', file4)
    rasterdsm = unir_raster2x2(file1dsm, file2dsm, file3dsm, file4dsm)
    rasterdem = unir_raster2x2(file1, file2, file3, file4)
    return rasterdem, rasterdsm


def obter_raster(ponto1, ponto2):  # (lon, lat)

    if ponto1[0] < 0:
        lon1 = str(int(np.ceil(-ponto1[0])))
        we1 = 'W'
    else:
        lon1 = str(int(np.floor(ponto1[0])))
        we1 = 'E'
    if ponto2[0] < 0:
        lon2 = str(int(np.ceil(-ponto2[0])))
        we2 = 'W'
    else:
        lon2 = str(int(np.floor(ponto2[0])))
        we2 = 'E'

    if ponto1[1] < 0:
        lat1 = str(int(np.ceil(-ponto1[1])))
        ns1 = 'S'
    else:
        lat1 = str(int(np.floor(ponto1[0])))
        ns1 = 'N'
    if ponto2[1] < 0:
        lat2 = str(int(np.ceil(-ponto2[1])))
        ns2 = 'S'
    else:
        lat2 = str(int(np.floor(ponto2[0])))
        ns2 = 'N'

    if len(lat1) == 2:
        raster1 = ns1 + lat1
        if ns1 == 'S':
            raster_landcover = ns1 + str(int(lat1) + (3 - (int(lat1) % 3)))
        else:
            raster_landcover = ns1 + str(int(lat1) - (int(lat1) % 3))
    else:
        raster1 = ns1 + '0' + lat1
        if ns1 == 'S':
            raster_landcover = ns1 + '0' + str(int(lat1) + (3 - (int(lat1) % 3)))
        else:
            raster_landcover = ns1 + '0' + str(int(lat1) - (int(lat1) % 3))
    if len(lat2) == 2:
        raster2 = ns2 + lat2
    else:
        raster2 = ns2 + '0' + lat2
    if len(lon1) == 3:
        raster1 = raster1 + we1 + lon1
        if we1 == 'W':
            raster_landcover = raster_landcover + we1 + str(int(lon1) + (3 - (int(lon1) % 3)))
        else:
            raster_landcover = raster_landcover + we1 + str(int(lon1) + - (int(lon1) % 3))
    elif len(lon1) == 2:
        raster1 = raster1 + we1 + '0' + lon1
        if we1 == 'W':
            raster_landcover = raster_landcover + we1 + '0' + str(int(lon1) + (3 - (int(lon1) % 3)))
        else:
            raster_landcover = raster_landcover + we1 + '0' + str(int(lon1) + - (int(lon1) % 3))
    else:
        raster1 = raster1 + we1 + '00' + lon1
        if we1 == 'W':
            raster_landcover = raster_landcover + we1 + '00' + str(int(lon1) + (3 - (int(lon1) % 3)))
        else:
            raster_landcover = raster_landcover + we1 + '00' + str(int(lon1) + - (int(lon1) % 3))
    if len(lon2) == 3:
        raster2 = raster2 + we2 + lon2
    elif len(lon2) == 2:
        raster2 = raster2 + we2 + '0' + lon2
    else:
        raster2 = raster2 + we2 + '00' + lon2

    if raster1 == raster2:
        return str(os.path.join('C:\PythonFlask\PlanCom\Raster', raster1 + '.tif')), str(
            os.path.join('C:\PythonFlask\PlanCom\dsm', raster1 + '.tif')), str(
            os.path.join('C:\PythonFlask\PlanCom\LandCover', raster_landcover + '.tif'))
    else:
        ref = ''
        if (we1 == 'W' and lon1 > lon2) or (we1 == 'E' and lon1 < lon2):
            ref = 'd'
        elif (we1 == 'W' and lon1 < lon2) or (we1 == 'E' and lon1 > lon2):
            ref = 'e'

        if (ns1 == 'S' and lat1 > lat2) or (we1 == 'N' and lat1 < lat2):
            ref = ref + 'c'
        elif (ns1 == 'S' and lat1 < lat2) or (we1 == 'N' and lat1 > lat2):
            ref = ref + 'b'

        raster, rasterdsm = unir_raster(raster1, raster2, ref)

        return raster, rasterdsm, str(
            os.path.join('C:\PythonFlask\PlanCom\LandCover', raster_landcover + '.tif'))


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
                # elevacao[i] = (1 + indice) * raster[int(vet_y[contmin])][int(vet_x[contmax])] - indice * raster[int(vet_y[contmin + 1])][int(vet_x[contmax])]
                xcoordg[i], ycoordg[i] = raster[int(vet_y[contmin + 1])][int(vet_x[contmax])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmax], vet_y[contmax + 1])
                contmin = contmin + 1
            contmax = contmax + 1
    else:
        for i in elevacao:
            i = int(i)
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
                # elevacao[i] = (1 + indice) * raster[int(vet_y[contmax])][int(vet_x[contmin])] - indice * raster[int(vet_y[contmax])][int(vet_x[contmin + 1])]
                elevacao[i] = raster[int(vet_y[contmax])][int(vet_x[contmin + 1])]
                xcoordg[i], ycoordg[i] = transform * (vet_x[contmin + 1], vet_y[contmax])
                contmin = contmin + 1

            contmax = contmax + 1
    return distancia, elevacao, xcoordg, ycoordg


def ajuste(elevacao, distancia, hg1, hg2, dl1, dl2):
    xa = int(min(15 * hg1, 0.1 * dl1) / distancia[1])
    xb = len(elevacao) - 1 - int(min(15 * hg2, 0.1 * dl2) / distancia[1])
    zorig = elevacao[xa:xb + 1]
    xorig = np.array(range(xa, xb + 1))
    z = []
    x = []
    u = 0

    # reduzir qtd de pontos usar quando for subamostrar
    while u <= len(xorig) - 5:
        xaux = np.mean(xorig[u:u + 4])
        zaux = np.mean(zorig[u:u + 4])
        x.append(xaux)
        z.append(zaux)
        u = u + 5
    z = np.array(zorig)
    x = np.array(xorig)

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

    try:
        c1 = ((syx / (n * s2x - (sx ** 2))) * n) - ((sy / (n * s2x - (sx ** 2))) * sx)
        c0 = ((s2x / (n * s2x - (sx ** 2))) * sy) - ((syx / (n * s2x - (sx ** 2))) * sx)
        zl = c1 * x + c0
        # calculo do he
        he1 = max(0, min(z[0], z[0] - zl[0])) + hg1
        he2 = max(0, min(z[-1], z[-1] - zl[-1])) + hg2
    except:
        he1 = z[0] + hg1
        he2 = z[-1] + hg2

    # calculo do Dh
    z = np.array(z)
    dados = z - zl
    # Calculando os percentis para dividir em 10 intervalos
    percentis = np.linspace(0, 100, 11)  # Divide em 10 partes iguais
    bins = np.percentile(dados, percentis)
    Dh = bins[-2] - bins[1]

    return he1, he2, Dh


def obter_dados_do_perfil(dem, dsm, distancia, ht, hr):
    angulo = []
    angulor = []
    demr = dem[::-1]
    d = distancia[-1]
    hg1, hg2 = ht, hr
    aref = np.arctan((-ht - dem[0] + hr + demr[0]) / d)
    visada = 1  # 'visada# '
    visadar = 1
    indice_visada_r = 0
    dl1, dl2, teta1, teta2 = 1e7, 1e7, None, None
    maxangulo = aref
    maxangulor = -aref

    for i in range(1, len(dem) - 1):
        angulo.append(np.arctan((dem[i] - (dem[0] + ht)) / distancia[i]))
        if (angulo[i - 1] > aref) and (angulo[i - 1] > maxangulo):
            teta1, dl1, idl1 = angulo[i - 1], distancia[i], i
            visada = 0
        maxangulo = max(angulo)

    for i in range(1, len(demr) - 1):
        angulor.append(np.arctan((demr[i] - (demr[0] + hr)) / distancia[i]))
        if (angulor[i - 1] > -aref) and (angulor[i - 1] > maxangulor):
            teta2, dl2, idl2 = angulor[i - 1], distancia[i], i
            visadar = 0
            indice_visada_r = len(demr) - (i + 1)
        maxangulor = max(angulor)

    visada = max(visada, visadar)

    he1, he2, Dh = ajuste(dem, distancia, hg1, hg2, dl1, dl2)
    # h é a altura dos telaho m
    # hb altura do transmissor, de 4 a 50- equivalente para cost25 sem visada
    h = max(0, np.mean(dsm[-3:len(dsm)]) - np.mean(dem[-3:len(dem)]))

    return d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h, visada, indice_visada_r


def get_land_surfelc_vec(loncoord, latcoord):  # loncoord=xcoor
    altura = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM')
    lcover10 = ee.ImageCollection("ESA/WorldCover/v200")
    dsm = []
    landcover = []
    for i in range(len(latcoord)):
        ponto = ee.Geometry.Point(loncoord[i], latcoord[i])

        dsm.append(altura.mean().sample(ponto, 30).first().get('DSM').getInfo())
        landcover.append(lcover10.first().sample(ponto, 10).first().get('Map').getInfo())
        if i < len(latcoord) - 1:
            latpasso = (latcoord[i + 1] - latcoord[i]) / 3
            lonpasso = (loncoord[i + 1] - loncoord[i]) / 3
            ponto2 = ee.Geometry.Point(loncoord[i] + lonpasso, latcoord[i] + latpasso)
            ponto3 = ee.Geometry.Point(loncoord[i] + 2 * lonpasso, latcoord[i] + 2 * latpasso)
            landcover.append(lcover10.first().sample(ponto2, 10).first().get('Map').getInfo())
            landcover.append(lcover10.first().sample(ponto3, 10).first().get('Map').getInfo())

    return landcover, dsm


def get_dados_landcover(indice, dem, landcover, dsm, hr, ht, distancia, h, dl2, visada, area=0):
    if visada:
        d_urb = distancia[-1]
    else:
        d_urb = dl2

    if landcover[-1] != 50:
        h = 0
    dem = np.array(dem)
    dsm = np.array(dsm)

    altur_da_cobertuta = dsm[indice:] - dem[indice:]
    espesura = 0
    # d_urb = ((len(dem) - indice) * distancia[1])
    if indice == 0:
        m = -(dem[indice] + ht - dem[-1] - hr) / distancia[-1]
        c = (dem[indice] + ht - dem[-1] - hr)
        x = np.array(distancia)
        y = m * x + c
        los = y - dem
        hb_urb = dem[0] - dem[-1] + ht
    else:
        m = -(dem[indice] - dem[-1] - hr) / ((len(dem) - (indice + 1)) * distancia[1])
        c = (dem[indice] - dem[-1] - hr)
        x = np.array(distancia[indice:])
        y = m * x + c
        los = y - dem[indice:]
        hb_urb = dem[indice] - dem[-1]

    for i in range(len(los) - 1):
        if los[i] < altur_da_cobertuta[i]:
            if area:
                if landcover[indice + i] == 10:
                    espesura = espesura + 10
            else:
                for n in (0, 1, 2):
                    if landcover[3 * indice + i + n] == 10:
                        espesura = espesura + 10  # ( colocar 5, metade dos 10 m)

    return espesura, h, d_urb, hb_urb


cobertura = []


def addfoliun():
    escala_de_altura = [0, 100]
    elvn = ee.Image("NASA/NASADEM_HGT/001")
    # ['00FFFF','00FFCC','33CCCC','669999','996699', 'CC3366', 'FF3366','FF0033','FF0000']
    image_viz_params = {'bands': ['elevation'], 'min': escala_de_altura[0], 'max': escala_de_altura[1],
                        'palette': ['0000ff', '00ffff', 'ffff00', 'ff0000', 'ffffff'],
                        'opacity': 0.7}  # Altura limitada par visualização BR

    # map_elevn = geemap.Map(center=[-22.9120, -43.2089], zoom=10)
    # map_elevn.add_layer(elvn, image_viz_params, 'Elevacao')

    folium_map = folium.Map(location=[-22.9120, -43.2089], zoom_start=7)

    folium.raster_layers.TileLayer(tiles='http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                                   attr='google',
                                   name='google maps',
                                   max_zoom=20,
                                   subdomains=['mt0', 'mt1', 'mt2', 'mt3'],
                                   overlay=False,
                                   control=True).add_to(folium_map)

    folium.raster_layers.TileLayer(tiles='http://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                                   attr='google',
                                   name='google maps street view',
                                   max_zoom=20,
                                   subdomains=['mt0', 'mt1', 'mt2', 'mt3'],
                                   overlay=False,
                                   control=True).add_to(folium_map)

    folium.raster_layers.TileLayer(tiles='http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                                   attr='google',
                                   name='google maps',
                                   max_zoom=20,
                                   subdomains=['mt0', 'mt1', 'mt2', 'mt3'],
                                   overlay=False,
                                   control=True).add_to(folium_map)

    map_id_dict = ee.Image(elvn).getMapId(image_viz_params)

    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name='elevacao',
        overlay=True,
        control=True
    ).add_to(folium_map)

    for i in cobertura:
        criamapa(i['raster'], i['img']).add_to(folium_map)

    """
    bdgex = 'http://bdgex.eb.mil.br/cgi-bin/mapaindice'
    bdgex_map = WebMapService(bdgex)
    print('\n'.join(bdgex_map.contents.keys()))
    layer = 'F100_WGS84_MATRICIAL'
    wms = bdgex_map.contents[layer]
    name = wms.title
    lon = (wms.boundingBox[0] + wms.boundingBox[2]) / 2
    lat = (wms.boundingBox[1] + wms.boundingBox[3]) / 2
    center = lat, lon
    style = 'boxfill/sst_36'

    if style not in wms.styles:
        style = None

    folium.raster_layers.WmsTileLayer(
        url=bdgex,
        name=name,
        style=style,
        fmt='image/png',
        transparent=False,
        layers=layer,
        overlay=True,
    ).add_to(folium_map)
    """
    folium_map.add_child(folium.LayerControl())
    return folium_map


markers = [{'lat': -22.9555, 'lon': -43.1661, 'nome': 'IME', 'h': 2.0},
           {'lat': -22.9036, 'lon': -43.1895, 'nome': 'PDC', 'h': 2.5}]


@app.route('/', methods=['GET', 'POST'])
def index():
    # tilesets cmt50: cmt50-wmsc-12, cmt100:cmt50-wmsc-24, curvas_nivel100: curvas_nivel100-wmsc-20, curvas_nivel50: curvas_nivel50-wmsc-8
    fol = addfoliun()
    global cobertura
    global markers

    fol.add_child(folium.LatLngPopup())
    # print(markers)
    for marker in markers:
        folium.Marker([marker['lat'], marker['lon']]).add_to(fol)
    map_file = 'map.html'
    map_file_path = os.path.join("templates", map_file)

    fol.save(map_file_path)

    return render_template('index1.html', map_file=map_file)


@app.route('/addponto', methods=['GET', 'POST'])
def addponto():
    return render_template('addponto.html')


@app.route('/add_marker', methods=['POST'])
def add_marker():
    # Obtemos as coordenadas do marcador do corpo da solicitação
    lat = float(request.form.get('lat'))
    lon = float(request.form.get('lon'))
    nome = str(request.form.get('nome'))
    h = float(request.form.get('h'))
    # Adicionamos o marcador à lista
    markers.append({'lat': lat, 'lon': lon, 'nome': nome, 'h': h})
    return jsonify({'result': 'success'})


perdas = []


@app.route('/ptp', methods=['GET', 'POST'])
def ptp():
    ht, hr = 0, 0
    p1 = ()
    p2 = ()
    if request.method == "POST":

        # calcular perda Aqui antes das operacoes abaixo
        if request.form.get("ponto1") and request.form.get("ponto2") and request.form.get("f"):

            for i in markers:
                if i['nome'] == request.form.get("ponto1"):
                    p1 = (i['lon'], i['lat'])
                    ht = i['h']
                elif i['nome'] == request.form.get("ponto2"):
                    p2 = (i['lon'], i['lat'])
                    hr = i['h']
            caminho, caminho_dsm, caminho_landcover = obter_raster(p1, p2)

            r = reta(p1, p2)
            f = float(request.form.get("f"))

            with rasterio.open(caminho) as raster:
                dem, dsm, landcover, distancia, dem_raster = perfil(r, raster)
                # distancia, dem, x, y = obter_perfil(p1, p2, raster)

            # landcover, dsm = get_land_surfelc_vec(x, y)

            d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h, visada, indice_visada_r = obter_dados_do_perfil(
                dem, dsm, distancia, ht, hr)
            if landcover[-1] == 50:
                urban = 'wi'
            else:
                urban = 'n'
            yt = 1  # é a perda pelo clima, adotar esse valor padrao inicialmente
            qs = 7  # 70% das situacões

            espesura, h, d_urb, hb_urb = get_dados_landcover(indice_visada_r, dem, landcover, dsm, hr, ht, distancia, h,
                                                             dl2, visada)
            # colocar a cidicao para chamar itm ou urbano + espaco livre

            print(
                f' ({f}, {hg1}, {hg2}, {he1}, {he2}, {d}, {yt}, {qs}, {dl1}, {dl2}, {Dh}, {visada}, {h},{teta1}, {teta2}, {d_urb}, {hb_urb}, {urban})')

            perda = itmModel.longLq_rice_model(f, hg1, hg2, he1, he2, d, yt, qs, dl1, dl2, Dh, visada, h,
                                               teta1, teta2, d_urb, hb_urb, urban, polarizacao='v')

            # colocar aqu uma funcao que adiciona a perda por vegetacao

            perdas.append({'ponto1': request.form.get("ponto1"),
                           'ponto2': request.form.get("ponto2"),
                           'f': f,
                           'perda': perda})

    return render_template('ptp.html', perdas=perdas)


@app.route('/area', methods=['GET', 'POST'])
def area():
    global cobertura
    p1 = ()
    ht = 2
    if request.form.get("ponto") and request.form.get("raio") and request.form.get("f"):
        limear = 100

        for i in markers:
            if i['nome'] == request.form.get("ponto"):
                p1 = (i['lon'], i['lat'])
                ht = i['h']

        caminho, caminho_dsm, caminho_landcover = obter_raster(p1, p1)
        caminho = modificar_e_salvar_raster(caminho, p1, float(request.form.get("raio")), limear, ht,
                                            float(request.form.get("f")))

        img = criaimg(caminho)
        cobertura.append(
            {'nome': request.form.get("ponto") + '_Area_de_cobertura' + '_' + request.form.get("f"), 'raster': caminho,
             'f': float(request.form.get("f")), 'img': img, 'h': ht})
        # aqui apenas criar a imagem, add ao mapa somente dento da funcao de ceiar mapa folium
        # escolher como nome da camada o nome do ponto+'Area de cobertura'
        # colocar a criamapa nesse arquivo dento dela chamar a fuçao de perda e definirvalor para fala ou não fala
    return render_template('area.html')


if __name__ == '__main__':
    app.run(debug=True)

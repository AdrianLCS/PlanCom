import numpy as np
import rasterio
from flask import Flask, render_template, request, jsonify
import os
import folium
import itmModel
import ee
import matplotlib.pyplot as plt
import Modelos

# correcao distancia 6228.6112900782355/5712.1356899878 ou 5719.711764799506

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




def parametros_difracao(distancia, dem, ht, hr):
    angulo = []
    demr = dem[::-1]
    d = distancia[-1]
    aref = np.arctan((-ht - dem[0] + hr + demr[0]) / d)
    visada = 1  # 'visada# '
    maxangulo = aref
    idl1 = 0
    dls = [0]
    hs = [ht+dem[0]]
    h, dl1, teta1 = 0,0,0
    for i in range(1, len(dem) - 1):
        angulo.append(np.arctan((dem[i] - (dem[0] + ht)) / distancia[i]))
        if (angulo[i - 1] > aref) and (angulo[i - 1] > maxangulo):
            dl1, idl1 = distancia[i], i
            h = dem[i]
            visada = 0
        maxangulo = max(angulo)
    if not visada:
        hs.append(h)
        dls.append(dl1)


    while not visada:
        angulo = []
        aref = np.arctan((- dem[idl1] - 1 + hr + demr[0]) / (d - distancia[idl1]))
        maxangulo=aref
        visada = 1
        for i in range(idl1 + 3, len(dem) - 1):
            angulo.append(np.arctan((dem[i] - (dem[idl1])) / (distancia[i] - distancia[idl1])))
            if (angulo[i - idl1 - 3] > aref) and (angulo[i - idl1 - 3] > maxangulo):
                dl1, idl1 = distancia[i], i
                h = dem[i]
                visada = 0
            maxangulo = max(angulo)
        if visada:
            break
        hs.append(h)
        dls.append(dl1)
    dls.append(d)
    hs.append(dem[-1]+hr)
    print(hs)
    print(dls)
    return dls, hs


def friis_free_space_loss_db(f, d):  # gt=direcionalidade*eficiencia
    comprimento_de_onda = c / (f * 1000000)
    L_db = -20 * np.log10(comprimento_de_onda) + 20 * np.log10(d) + 22
    return L_db  # Esse valor é de fato uma perda


def calcula_perda(ht, hr, f, r, raster, raster_dsm, raster_landcover):
    dem, dsm, landcover, distancia = perfil(r, raster, raster_dsm, raster_landcover)

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

    # print(
    #    f' ({f}, {hg1}, {hg2}, {he1}, {he2}, {d}, {yt}, {qs}, {dl1}, {dl2}, {Dh}, {visada}, {h},{teta1}, {teta2}, {d_urb}, {hb_urb}, {urban})')
    perda = itmModel.longLq_rice_model(f, hg1, hg2, he1, he2, d, yt, qs, dl1, dl2, Dh, visada, h,
                                       teta1, teta2, d_urb, hb_urb, urban, polarizacao='v')

    return perda


def modificar_e_salvar_raster(raster_path, caminho_dsm, caminho_landcover, ponto, raio, limear, ht, f):
    # dem = ee.Image("NASA/NASADEM_HGT/001")
    # dsm = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM')
    # lcover10 = ee.ImageCollection("ESA/WorldCover/v200")

    pasta = raster_path[:-11] + 'modificado'
    file = '\A' + raster_path[-11:]
    # Abrir o arquivo raster para leitura e escrita
    with rasterio.open(raster_path, 'r+') as src, rasterio.open(caminho_dsm) as raster_dsm, rasterio.open(
            caminho_landcover) as raster_landcover:
        # Ler a matriz de dados do raster
        data = src.read(1)
        inv_transform = ~src.transform
        transform = src.transform
        x, y = inv_transform * (ponto[0], ponto[1])
        unidade_distancia = transform[0] * (10 ** 5)
        raio = raio / unidade_distancia

        # Modificar o valor do ponto desejado
        for linha in range(np.shape(data)[0]):
            for coluna in range(np.shape(data)[1]):
                if (((((linha - y) ** 2) + ((coluna - x) ** 2)) ** 0.5) < (raio - 2)) and (
                        ((((linha - y) ** 2) + ((coluna - x) ** 2)) ** 0.5) * unidade_distancia > 200):
                    p2lon, p2lat = transform * (coluna, linha)
                    r = reta(ponto, (p2lon, p2lat))
                    p = calcula_perda(ht, 2, f, r, src, raster_dsm, raster_landcover)
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
    n = int(np.ceil(modulo / 0.000277777777777778))  # precisao de 30 m
    t = np.linspace(0, 1, n)
    r = []
    for i in t:
        r.append(p1 + v * i)
    r = np.array(r)
    return r


a = 6378137
b = 6356752


def R(lat):
    return (((((a ** 2) * np.cos(lat * np.pi / 180)) ** 2) + (((b ** 2) * np.sin(lat * np.pi / 180)) ** 2)) / (
            ((a * np.cos(lat * np.pi / 180)) ** 2) + ((b * np.sin(lat * np.pi / 180)) ** 2))) ** 0.5


def perfil(r, src, src_dsm, src_landcover):
    raster = src.read(1)
    raster_dsm = src_dsm.read(1)
    raster_landcover = src_landcover.read(1)
    inv_transform = ~src.transform
    inv_transform_dsm = ~src_dsm.transform
    inv_transform_landcover = ~src_landcover.transform
    unidade_distancia = 2 * np.pi * R(r[0][1]) / (1296000)
    x0, y0 = inv_transform * (r[0][0], r[0][1])
    dem = []
    dsm = []
    landcover = []
    pixel_xn, pixel_yn = inv_transform * (r[np.shape(r)[0] - 1][0], r[np.shape(r)[0] - 1][1])
    distancia = unidade_distancia * ((((pixel_xn - x0) ** 2) + ((pixel_yn - y0) ** 2)) ** 0.5) / (np.shape(r)[0] - 1)
    d = []
    for i in range(np.shape(r)[0]):
        pixel_x1, pixel_y1 = inv_transform * (r[i][0], r[i][1])
        pixel_x1_dsm, pixel_y1_dsm = inv_transform_dsm * (r[i][0], r[i][1])
        pixel_x1_lancover, pixel_y1_landcover = inv_transform_landcover * (r[i][0], r[i][1])
        dist = distancia * i
        alt_dem = ((1 - (pixel_y1 - np.floor(pixel_y1))) * raster[int(np.floor(pixel_y1))][int(np.floor(pixel_x1))] + (
                pixel_y1 - np.floor(pixel_y1)) * raster[int(np.ceil(pixel_y1))][int(np.floor(pixel_x1))] + (
                           1 - (pixel_x1 - np.floor(pixel_x1))) * raster[int(np.floor(pixel_y1))][
                       int(np.floor(pixel_x1))] + (
                           pixel_x1 - np.floor(pixel_x1)) * raster[int(np.floor(pixel_y1))][int(np.ceil(pixel_x1))]) / 2

        alt_dsm = ((1 - (pixel_y1_dsm - np.floor(pixel_y1_dsm))) * raster_dsm[int(np.floor(pixel_y1_dsm))][
            int(np.floor(pixel_x1_dsm))] + (
                           pixel_y1 - np.floor(pixel_y1_dsm)) * raster_dsm[int(np.ceil(pixel_y1_dsm))][
                       int(np.floor(pixel_x1_dsm))] + (
                           1 - (pixel_x1_dsm - np.floor(pixel_x1_dsm))) * raster_dsm[int(np.floor(pixel_y1_dsm))][
                       int(np.floor(pixel_x1_dsm))] + (
                           pixel_x1_dsm - np.floor(pixel_x1_dsm)) * raster_dsm[int(np.floor(pixel_y1_dsm))][
                       int(np.ceil(pixel_x1_dsm))]) / 2

        d.append(dist)
        dem.append(alt_dem)
        dsm.append(alt_dsm)
        landcover.append(raster_landcover[int(pixel_y1_landcover)][int(pixel_x1_lancover)])
        if i < np.shape(r)[0] - 1:
            lonpasso = (r[i + 1][0] - r[i][0]) / 3
            latpasso = (r[i + 1][1] - r[i][1]) / 3
            pixel_x2_lancover, pixel_y2_landcover = inv_transform_landcover * (r[i][0] + lonpasso, r[i][1] + latpasso)
            pixel_x3_lancover, pixel_y3_landcover = inv_transform_landcover * (
                r[i][0] + 2 * lonpasso, r[i][1] + 2 * latpasso)
            landcover.append(raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)])
            landcover.append(raster_landcover[int(pixel_y3_landcover)][int(pixel_x3_lancover)])

    return dem, dsm, landcover, d


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
        zl = 0

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
    dl1, dl2, teta1, teta2 = d, d, None, None
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
    h = max(0, 1.5 * np.mean(dsm[-4:len(dsm)]) - np.mean(dem[-4:len(dem)]))

    return d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h, visada, indice_visada_r


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
        los = y - (dem - (dem[-1] + hr))
        hb_urb = dem[0] - dem[-1] + ht
    else:
        rfresn = 0.6 * Modelos.raio_fresnel(1, distancia[indice], distancia[-1] - distancia[indice], f)
        m = -(rfresn + dem[indice] - dem[-1] - hr) / ((len(dem) - (indice + 1)) * distancia[1])
        c = (rfresn + dem[indice] - dem[-1] - hr)
        x = np.array(distancia[indice:])
        x = x - distancia[indice]
        y = m * x + c
        los = y - (dem[indice:] - (dem[-1] + hr))
        hb_urb = dem[indice] - dem[-1]

    for i in range(len(los) - 1):
        if los[i] < altur_da_cobertuta[i]:
            for n in (0, 1, 2):
                if landcover[3 * indice + i + n] == 10:
                    espesura = espesura + 10  # ( colocar 5, metade dos 10 m)

    return espesura, h, d_urb, hb_urb


cobertura = []
markers = [{'lat': -22.9555, 'lon': -43.1661, 'nome': 'IME', 'h': 2.0},
           {'lat': -22.9036, 'lon': -43.1895, 'nome': 'PDC', 'h': 22.5}]

p1 = (markers[1]['lon'], markers[1]['lat'])
p2 = (markers[0]['lon'], markers[0]['lat'])
caminho, caminho_dsm, caminho_landcover = obter_raster(p1, p2)

r = reta(p1, p2)
f = float(800)
ime = 2
PDC = 22
with rasterio.open(caminho) as raster, rasterio.open(caminho_dsm) as raster_dsm, rasterio.open(
        caminho_landcover) as raster_landcover:
    dem, dsm, landcover, distancia = perfil(r, raster, raster_dsm, raster_landcover)

d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h, visada, indice_visada_r = obter_dados_do_perfil(
    dem, dsm, distancia, PDC, ime)
if landcover[-1] == 50:
    urban = 'wi'
else:
    urban = 'n'
yt = 1  # é a perda pelo clima, adotar esse valor padrao inicialmente
qs = 7  # 70% das situacões
espesura, h, d_urb, hb_urb = get_dados_landcover(indice_visada_r, dem, landcover, dsm, ime, PDC, distancia, h,
                                                 dl2, visada)
# colocar a cidicao para chamar itm ou urbano + espaco livre

print(
    f' ({f}, {hg1}, {hg2}, {he1}, {he2}, {d}, {yt}, {qs}, {dl1}, {dl2}, {Dh}, {visada}, {h},{teta1}, {teta2}, {d_urb}, {hb_urb}, {urban})')

h0 = (dem[0] + dem[-1]) / 2

dls,hs = parametros_difracao(distancia, dem, hg1, hg2)
perda_difrac = Modelos.modelo_epstein_peterson(dls, hs,f)

perda = Modelos.longLq_rice_model(h0, f, hg1, hg2, he1, he2, d, yt, qs, dl1, dl2, Dh, visada,
                                  teta1, teta2, polarizacao='v', simplificado=0)

espaco_livre = Modelos.friis_free_space_loss_db(f, d)

if urban == 'wi' and h > hg2 + 0.5:
    urb = Modelos.ikegami_model(h, hg2, f)
else:
    urb = 0

vegetacao = Modelos.atenuaca_vegetacao_antiga_ITU(f, espesura)
print(espaco_livre)

print(perda)

print(urb)
print(vegetacao)
print(espesura)
print(perda_difrac)
#plt.plot(distancia, dem)
#plt.title('Modelo Digital de Elevação (DEM)')

#plt.show()


a=np.array((1,2))
b = np.array([3.1,4])
print(int(a+b))
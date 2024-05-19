import numpy as np
import rasterio
from flask import Flask, render_template, request, jsonify
import os
import folium
import matplotlib.pyplot as plt
import Modelos
from numpy import sin, cos, arccos, pi, round


def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians


def getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2):
    theta = longitude1 - longitude2
    latitude1=deg2rad(latitude1)
    latitude2 = deg2rad(latitude2)
    longitude1 = deg2rad(longitude1)
    longitude2 = deg2rad(longitude2)
    distance = 2*R((latitude1+latitude2)/2) * np.arcsin(((np.sin((latitude2-latitude1)/2))**2+
                                                         np.cos(latitude1)*np.cos(latitude2)*((np.sin((longitude2-longitude1)/2))**2))**0.5)
    return distance


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
app = Flask(__name__)
c = 299792458  # m/s
a = 6378137  # m
b = 6356752  # m

Configuracao = {"modelo": "ITM", "urb": 1, "veg": 1, "precisao": 0.5, "max_alt": 300,
                "min_alt": 0}  # ITM ou Epstein-peterson
mapas = []


def extrair_vet_area(raio, ponto, f, limear, unidade_distancia, precisao):
    comprimento_de_onda = c / (f * 1000000)
    # L_db = -20 * np.log10(comprimento_de_onda) + 20 * np.log10(d) + 22
    d = 10 ** ((limear / 20) + np.log10(comprimento_de_onda) - 1.1)
    d = min(raio, d)
    print(d)
    retas = []
    dem0, dsm0, landcover0, distancia0 = [], [], [], []
    for i in range(int(360 * precisao)):
        vet = np.array([np.cos(i * 2 * np.pi / (precisao * 360)), np.sin(
            i * 2 * np.pi / (precisao * 360))])  # roda no sentido positivo trigonométrio de 0.5 em 0.5 graus
        pf = np.array(ponto) + vet * (
                d / unidade_distancia) * (1 / 3600)
        print(ponto)
        print(pf)
        dem, dsm, landcover, distancia, r = perfil(ponto, pf, 1)
        distancia0.append(distancia)
        retas.append(r)
        dem0.append(dem)
        dsm0.append(dsm)
        landcover0.append(landcover)
    print('criou as retas')
    return retas, d, dem0, dsm0, landcover0, distancia0


def parametros_difracao(distancia, dem, ht, hr, frequ):
    angulo = []
    d = distancia[-1]
    aref = (hr + dem[-1] -ht - dem[0]) / d
    visada = 1  # 'visada# '
    maxangulo = aref
    dls = [0]
    hs = [ht + dem[0]]
    h, idl1, teta1 = 0, 0, 0
    for i in range(1, len(dem) - 1):
        if i > 6 and i < len(dem) - 6:
            rfresn2 = 0.6 * Modelos.raio_fresnel(1, distancia[i], distancia[-1] - distancia[i], frequ)
        else:
            rfresn2 = 0
        angulo.append((dem[i] + rfresn2 - (dem[0] + ht)) / distancia[i])
        if angulo[-1] > maxangulo:
            idl1 = i
            h = dem[i]
            visada = 0
            maxangulo = max(angulo)
    if not visada:
        hs.append(h)
        dls.append(distancia[idl1])

    while not visada:
        idll=[idl1]
        angulo = []
        aref = ( hr + dem[-1] - dem[idl1]) / (d - distancia[idl1])
        maxangulo = aref
        visada = 1
        for i in range(idl1 + 5, len(dem) - 1):
            angulo.append((dem[i] - (dem[idl1])) / (distancia[i] - distancia[idl1]))
            if  (angulo[-1] > maxangulo):
                idll.append(i)
                h = dem[i]
                visada = 0
                maxangulo = max(angulo)
        idl1 = idll[-1]
        if visada:
            break
        hs.append(h)
        dls.append(distancia[idl1])
    dls.append(d)
    hs.append(dem[-1] + hr)
    return dls, hs


def modificar_e_salvar_raster(raster_path, ponto, raio, limear, ht, hr, f, precisao):
    pasta = raster_path[:-11] + 'modificado'
    file = '\A' + raster_path[-11:]
    yt = 1
    qs = 5
    unidade_distancia = 2 * np.pi * R(ponto[1]) / (1296000)
    retas, raio, dem0, dsm0, landcover0, distancia0 = extrair_vet_area(raio, ponto, f, limear, unidade_distancia,
                                                                       precisao)
    # Abrir o arquivo raster para leitura e escrita
    with rasterio.open(raster_path, 'r+') as src:
        # Ler a matriz de dados do raster
        data = src.read(1)
        inv_transform = ~src.transform
        x, y = inv_transform * (ponto[0], ponto[1])

        # Modificar o valor do ponto desejado
        for linha in range(np.shape(data)[0]):
            for coluna in range(np.shape(data)[1]):
                distyx = ((((linha - y) ** 2) + ((coluna - x) ** 2)) ** 0.5)
                if (distyx * unidade_distancia > 200) and (distyx < ((raio / unidade_distancia) - 3)):

                    if coluna != x:
                        if coluna > x:
                            angulo = np.arctan((y - linha) / (coluna - x))
                        else:
                            angulo = np.arctan((y - linha) / (coluna - x)) + np.pi
                    elif y > linha:
                        angulo = np.pi / 2
                    else:
                        angulo = 3 * np.pi / 2

                    if angulo < 0:
                        angulo = 2 * np.pi + angulo
                    angulo2 = int((180 * angulo / np.pi) * precisao)
                    r = retas[angulo2][:int(distyx + 1)]
                    dem = dem0[angulo2][:int(distyx + 1)]
                    dsm = dsm0[angulo2][:int(distyx + 1)]
                    landcover = landcover0[angulo2][:3 * int(distyx) + 1]
                    distancia = distancia0[angulo2][:int(distyx + 1)]

                    Densidade_urbana = 0.7
                    d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h_urb, visada, indice_visada_r, indice_visada = obter_dados_do_perfil(
                        dem, dsm, distancia, ht, hr, Densidade_urbana)
                    hmed = (dem[0] + dem[-1]) / 2

                    if visada:
                        data[linha][coluna] = 2
                    else:
                        if (landcover[-1] == 50) and (h_urb > hg2 + 0.5):
                            urb = Modelos.ikegami_model(h_urb, hg2, f)
                        else:
                            urb = 0
                        perda, variabilidade_situacao, At, dls_LR = Modelos.longLq_rice_model(hmed, f, hg1, hg2, he1, he2, d,
                                                                                      yt, qs, dl1,
                                                                                      dl2,
                                                                                      Dh, visada,
                                                                                      teta1, teta2, polarizacao='v')

                        espaco_livre = Modelos.friis_free_space_loss_db(f, d)
                        espesura = obter_vegeta_atravessada(f, indice_visada_r, dem, landcover, dsm, hr, ht, distancia,
                                                            indice_visada)
                        vegetacao = Modelos.atenuaca_vegetacao_antiga_ITU(f, espesura)
                        p = espaco_livre + perda + variabilidade_situacao + urb + vegetacao

                        if p <= limear:
                            data[linha][coluna] = 2
                        else:
                            data[linha][coluna] = 100
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


def reta(p1, p2, tranform):
    global a
    p1 = np.array(p1)
    p2 = np.array(p2)
    v = p2 - p1

    modulo = np.linalg.norm(v)
    n = int(np.ceil(modulo / tranform))  # precisao de 30 m
    t = np.linspace(0, 1, n)
    r = []
    unidade_dist=(modulo*np.pi * a / 180)/(n-1)
    for i in t:
        r.append(p1 + v * i)
    r = np.array(r)
    dist = getDistanceBetweenPointsNew(p1[1], p1[0], p2[1], p2[0])
    unidade_dist = dist / (n - 1)
    return r, unidade_dist


def R(lat):
    global a
    global b
    return (((((a ** 2) * np.cos(lat * np.pi / 180)) ** 2) + (((b ** 2) * np.sin(lat * np.pi / 180)) ** 2)) / (
            ((a * np.cos(lat * np.pi / 180)) ** 2) + ((b * np.sin(lat * np.pi / 180)) ** 2))) ** 0.5


def obter_dados_do_raster2(indice_atual, r, dem, dsm, landcover, d, distancia, area):
    caminho, caminho_dsm, caminho_landcover = obter_raster(r[indice_atual], r[indice_atual])
    print(r[indice_atual])
    print(caminho)
    global Configuracao
    if (Configuracao["urb"] or Configuracao["veg"]) or not area:
        with rasterio.open(caminho) as src:
            raster = src.read(1)
            inv_transform = ~src.transform
            indice_atual_dem = indice_atual
            for i in range(indice_atual, np.shape(r)[0]):
                if (np.floor(r[i][0]) == np.floor(r[indice_atual_dem][0])) and (
                        np.floor(r[i][1]) == np.floor(r[indice_atual_dem][1])):
                    pixel_x1, pixel_y1 = inv_transform * (r[i][0], r[i][1])
                    dist = distancia * i
                    alt_dem = raster[int(pixel_y1)][int(pixel_x1)]

                    d.append(dist)
                    dem.append(alt_dem)
                    indice_atual_dem = i
                else:
                    indice_atual_dem = i
                    break
        with rasterio.open(caminho_dsm) as src_dsm:
            raster_dsm = src_dsm.read(1)
            inv_transform_dsm = ~src_dsm.transform

            indice_atual_dsm = indice_atual
            for i in range(indice_atual, np.shape(r)[0]):
                if (np.floor(r[i][0]) == np.floor(r[indice_atual_dsm][0])) and (
                        np.floor(r[i][1]) == np.floor(r[indice_atual_dsm][1])):
                    pixel_x1_dsm, pixel_y1_dsm = inv_transform_dsm * (r[i][0], r[i][1])

                    alt_dsm = raster_dsm[int(pixel_y1_dsm)][int(pixel_x1_dsm)]
                    dsm.append(alt_dsm)
                    indice_atual_dsm = i
                else:
                    break

        with rasterio.open(caminho_landcover) as src_landcover:
            raster_landcover = src_landcover.read(1)
            inv_transform_landcover = ~src_landcover.transform
            indice_atual_land = indice_atual
            for i in range(indice_atual, np.shape(r)[0]):
                if (np.floor(r[i][0]) == np.floor(r[indice_atual_land][0])) and (
                        np.floor(r[i][1]) == np.floor(r[indice_atual_land][1])):
                    pixel_x1_lancover, pixel_y1_landcover = inv_transform_landcover * (r[i][0], r[i][1])
                    landcover.append(raster_landcover[int(pixel_y1_landcover)][int(pixel_x1_lancover)])
                    if i < np.shape(r)[0] - 1:
                        lonpasso = (r[i + 1][0] - r[i][0]) / 3
                        latpasso = (r[i + 1][1] - r[i][1]) / 3
                        pixel_x2_lancover, pixel_y2_landcover = inv_transform_landcover * (
                            r[i][0] + lonpasso, r[i][1] + latpasso)
                        pixel_x3_lancover, pixel_y3_landcover = inv_transform_landcover * (
                            r[i][0] + 2 * lonpasso, r[i][1] + 2 * latpasso)
                        landcover.append(raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)])
                        landcover.append(raster_landcover[int(pixel_y3_landcover)][int(pixel_x3_lancover)])
                    indice_atual_land = i
                else:
                    break
        indice_atual = indice_atual_dem
        return dem, dsm, landcover, d, indice_atual

    else:
        with rasterio.open(caminho) as src:
            raster = src.read(1)
            inv_transform = ~src.transform
            for i in range(np.shape(r)[0]):
                if (np.floor(r[i][0]) == np.floor(r[indice_atual][0])) and (
                        np.floor(r[i][1]) == np.floor(r[indice_atual][1])):
                    pixel_x1, pixel_y1 = inv_transform * (r[i][0], r[i][1])
                    dist = distancia * i

                    alt_dem = raster[int(pixel_y1)][int(pixel_x1)]

                    d.append(dist)
                    dem.append(alt_dem)
                    indice_atual = i
                else:
                    indice_atual = i
                    break
        return dem, dsm, landcover, d, indice_atual


def obter_dados_do_raster(indice_atual, r, dem, dsm, landcover, d, distancia, area):
    caminho, caminho_dsm, caminho_landcover = obter_raster(r[indice_atual], r[indice_atual])
    print(r[indice_atual])
    print(caminho)
    global Configuracao
    if (Configuracao["urb"] or Configuracao["veg"]) or not area:
        with rasterio.open(caminho) as src:
            raster = src.read(1)
            inv_transform = ~src.transform
            indice_atual_dem = indice_atual
            for i in range(indice_atual, np.shape(r)[0]):
                if (np.floor(r[i][0]) == np.floor(r[indice_atual_dem][0])) and (
                        np.floor(r[i][1]) == np.floor(r[indice_atual_dem][1])):
                    pixel_x1, pixel_y1 = inv_transform * (r[i][0], r[i][1])
                    dist = distancia * i
                    alt_dem = raster[int(pixel_y1)][int(pixel_x1)]

                    d.append(dist)
                    dem.append(alt_dem)
                    indice_atual_dem = i
                else:
                    indice_atual_dem = i
                    break
        with rasterio.open(caminho_dsm) as src_dsm:
            raster_dsm = src_dsm.read(1)
            inv_transform_dsm = ~src_dsm.transform

            indice_atual_dsm = indice_atual
            for i in range(indice_atual, np.shape(r)[0]):
                if (np.floor(r[i][0]) == np.floor(r[indice_atual_dsm][0])) and (
                        np.floor(r[i][1]) == np.floor(r[indice_atual_dsm][1])):
                    pixel_x1_dsm, pixel_y1_dsm = inv_transform_dsm * (r[i][0], r[i][1])

                    alt_dsm = raster_dsm[int(pixel_y1_dsm)][int(pixel_x1_dsm)]
                    dsm.append(alt_dsm)
                    indice_atual_dsm = i
                else:
                    break

        with rasterio.open(caminho_landcover) as src_landcover:
            raster_landcover = src_landcover.read(1)
            inv_transform_landcover = ~src_landcover.transform
            indice_atual_land = indice_atual
            for i in range(indice_atual, np.shape(r)[0]):
                if (np.floor(r[i][0]) == np.floor(r[indice_atual_land][0])) and (
                        np.floor(r[i][1]) == np.floor(r[indice_atual_land][1])):
                    pixel_x1_lancover, pixel_y1_landcover = inv_transform_landcover * (r[i][0], r[i][1])
                    landcover.append(raster_landcover[int(pixel_y1_landcover)][int(pixel_x1_lancover)])
                    if i < np.shape(r)[0] - 1:
                        lonpasso = (r[i + 1][0] - r[i][0]) / 3
                        latpasso = (r[i + 1][1] - r[i][1]) / 3
                        pixel_x2_lancover, pixel_y2_landcover = inv_transform_landcover * (
                            r[i][0] + lonpasso, r[i][1] + latpasso)
                        pixel_x3_lancover, pixel_y3_landcover = inv_transform_landcover * (
                            r[i][0] + 2 * lonpasso, r[i][1] + 2 * latpasso)
                        if (np.floor(r[i][0] + 2*lonpasso) == np.floor(r[indice_atual_land][0])) and (
                                np.floor(r[i][1] + 2*latpasso) == np.floor(r[indice_atual_land][1])):
                            landcover.append(raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)])
                            landcover.append(raster_landcover[int(pixel_y3_landcover)][int(pixel_x3_lancover)])
                        elif (np.floor(r[i][0] + lonpasso) == np.floor(r[indice_atual_land][0])) and (
                                np.floor(r[i][1] + latpasso) == np.floor(r[indice_atual_land][1])):
                            landcover.append(raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)])
                            landcover.append(raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)])
                        else:
                            landcover.append(raster_landcover[int(pixel_y1_landcover)][int(pixel_x1_lancover)])
                            landcover.append(raster_landcover[int(pixel_y1_landcover)][int(pixel_x1_lancover)])
                    indice_atual_land = i
                else:
                    break
        indice_atual = indice_atual_dem
        return dem, dsm, landcover, d, indice_atual
    else:

        with rasterio.open(caminho) as src:
            raster = src.read(1)
            inv_transform = ~src.transform
            for i in range(np.shape(r)[0]):
                if (np.floor(r[i][0]) == np.floor(r[indice_atual][0])) and (
                        np.floor(r[i][1]) == np.floor(r[indice_atual][1])):
                    pixel_x1, pixel_y1 = inv_transform * (r[i][0], r[i][1])
                    dist = distancia * i

                    alt_dem = raster[int(pixel_y1)][int(pixel_x1)]

                    d.append(dist)
                    dem.append(alt_dem)
                    indice_atual = i
                else:
                    indice_atual = i
                    break
        return dem, dsm, landcover, d, indice_atual


def perfil(p1, p2, area=0):
    indice_atual = 0
    dem = []
    dsm = []
    landcover = []
    d = []
    caminho, caminho_dsm, caminho_landcover = obter_raster(p1, p1)
    with rasterio.open(caminho) as src1:
        transform = src1.transform
        r, distancia = reta(p1, p2, transform[0])
    while indice_atual < np.shape(r)[0] - 1:
        dem, dsm, landcover, d, indice_atual = obter_dados_do_raster(indice_atual, r, dem, dsm, landcover, d, distancia,
                                                                     area)

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

    file1 = os.path.join('Raster', file1)
    file2 = os.path.join('Raster', file2)
    file3 = os.path.join('Raster', file3)
    file4 = os.path.join('Raster', file4)
    file1dsm = os.path.join('dsm', file1)
    file2dsm = os.path.join('dsm', file2)
    file3dsm = os.path.join('dsm', file3)
    file4dsm = os.path.join('dsm', file4)
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
        lat1 = str(int(np.floor(ponto1[1])))
        ns1 = 'N'
    if ponto2[1] < 0:
        lat2 = str(int(np.ceil(-ponto2[1])))
        ns2 = 'S'
    else:
        lat2 = str(int(np.floor(ponto2[1])))
        ns2 = 'N'

    if len(lat1) == 2:
        raster1 = ns1 + lat1
        if ns1 == 'S':
            if (int(lat1) % 3) == 0:
                raster_landcover = ns1 + str(int(lat1))
            else:
                raster_landcover = ns1 + str(int(lat1) + (3 - (int(lat1) % 3)))
        else:
            lat1_land = str(int(lat1) - (int(lat1) % 3))
            if len(lat1_land) == 2:
                raster_landcover = ns1 + lat1_land
            else:
                raster_landcover = ns1 + '0' + lat1_land
    else:
        raster1 = ns1 + '0' + lat1
        if (int(lat1) % 3) == 0:
            lat1_land = str(int(lat1))
        else:
            lat1_land = str(int(lat1) + (3 - (int(lat1) % 3)))
        if ns1 == 'S':
            if len(lat1_land) == 2:
                raster_landcover = ns1 + lat1_land
            else:
                raster_landcover = ns1 + '0' + lat1_land
        else:
            raster_landcover = ns1 + '0' + str(int(lat1) - (int(lat1) % 3))


    if len(lon1) == 3:
        raster1 = raster1 + we1 + lon1
        if we1 == 'W':
            if (int(lon1) % 3)==0:
                raster_landcover = raster_landcover + we1 + str(int(lon1))
            else:
                raster_landcover = raster_landcover + we1 + str(int(lon1) + (3 - (int(lon1) % 3)))
        else:
            lon1_land = str(int(lon1) - (int(lon1) % 3))
            if len(lon1_land)==3:
                raster_landcover = raster_landcover + we1 + lon1_land
            else:
                raster_landcover = raster_landcover + we1 + '0' + lon1_land
    elif len(lon1) == 2:
        raster1 = raster1 + we1 + '0' + lon1
        if we1 == 'W':
            if (int(lon1) % 3) == 0:
                raster_landcover = raster_landcover + we1 +'0'+ str(int(lon1))
            else:
                raster_landcover = raster_landcover + we1 +'0'+ str(int(lon1) + (3 - (int(lon1) % 3)))

        else:
            lon1_land = str(int(lon1) - (int(lon1) % 3))
            if len(lon1_land) == 2:
                raster_landcover = raster_landcover + we1 + '0' + lon1_land
            else:
                raster_landcover = raster_landcover + we1 + '00' + lon1_land

    else:
        raster1 = raster1 + we1 + '00' + lon1
        if we1 == 'W':
            if (int(lon1) % 3) == 0:
                raster_landcover = raster_landcover + we1 +'00'+ str(int(lon1))
            else:
                raster_landcover = raster_landcover + we1 +'00'+ str(int(lon1) + (3 - (int(lon1) % 3)))
        else:
            raster_landcover = raster_landcover + we1 + '00' + str(int(lon1) + - (int(lon1) % 3))


    if len(lat2) == 2:
        raster2 = ns2 + lat2
    else:
        raster2 = ns2 + '0' + lat2
    if len(lon2) == 3:
        raster2 = raster2 + we2 + lon2
    elif len(lon2) == 2:
        raster2 = raster2 + we2 + '0' + lon2
    else:
        raster2 = raster2 + we2 + '00' + lon2

    if raster1 == raster1:
        return str(os.path.join('Raster', raster1 + '.tif')), str(
            os.path.join('dsm', raster1 + '.tif')), str(
            os.path.join('LandCover', raster_landcover + '.tif'))
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
            os.path.join('LandCover', raster_landcover + '.tif'))


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
    z = np.array(z)#np.array(zorig)
    x = np.array(x)#np.array(xorig)

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


def obter_dados_do_perfil(dem, dsm, distancia, ht, hr, Densidade_urbana):
    angulo = []
    angulor = []
    demr = dem[::-1]
    d = distancia[-1]
    hg1, hg2 = ht, hr
    aref = np.arctan((-ht - dem[0] + hr + demr[0]) / d)
    visada = 1  # 'visada# '
    visadar = 1
    indice_visada_r = 0
    indice_visada = 0
    dl1, dl2, teta1, teta2 = d, d, None, None
    maxangulo = aref
    maxangulor = -aref

    for i in range(1, len(dem) - 1):
        angulo.append(np.arctan((dem[i] - (dem[0] + ht)) / distancia[i]))
        if angulo[-1] > maxangulo:
            teta1, dl1, idl1 = angulo[i - 1], distancia[i], i
            visada = 0
            indice_visada = idl1
            maxangulo = max(angulo)

    for i in range(1, len(demr) - 1):
        angulor.append(np.arctan((demr[i] - (demr[0] + hr)) / distancia[i]))
        if angulor[-1] > maxangulor:
            teta2, dl2, idl2 = angulor[i - 1], distancia[i], i
            visadar = 0
            indice_visada_r = len(demr) - (i + 1)
            maxangulor = max(angulor)
    visada = max(visada, visadar)

    he1, he2, Dh = ajuste(dem, distancia, hg1, hg2, dl1, dl2)
    # h é a altura dos telaho m
    # hb altura do transmissor, de 4 a 50- equivalente para cost25 sem visada
    h_urb = abs((1 / Densidade_urbana) * (dsm[-1] - dem[-1]))

    return d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h_urb, visada, indice_visada_r, indice_visada


def obter_vegeta_atravessada(f, indice, dem, landcover, dsm, hr, ht, distancia, indice_d):
    dem = np.array(dem)
    dsm = np.array(dsm)

    altur_da_cobertuta = abs(dsm[indice:] - dem[indice:])
    espesura = 0
    if indice == 0:
        m = -(dem[0] + ht - dem[-1] - hr) / distancia[-1]
        c = (dem[0] + ht - dem[-1] - hr)
        x = np.array(distancia)
        if c < 0:
            y = m * x
            los = y - (dem - (dem[0] + ht))
        else:
            y = m * x + c
            los = y - (dem - (dem[-1] + hr))
        for i in range(len(los) - 1):
            if los[i] < altur_da_cobertuta[i]:
                for n in (0, 1, 2):
                    if landcover[3 * (indice + i) + n] == 10:
                        espesura = espesura + 10  # ( colocar 5, metade dos 10 m)


    else:
        indice_d = max(1, indice_d)
        rfresn = 0.6 * Modelos.raio_fresnel(1, distancia[indice], distancia[-1] - distancia[indice], f)
        m = -(rfresn + dem[indice] - dem[-1] - hr) / ((len(dem) - (indice + 1)) * distancia[1])
        c = (rfresn + dem[indice] - dem[-1] - hr)
        x = np.array(distancia[indice:])
        x = x - distancia[indice]
        if c < 0:
            y = m * x
            los = y - (dem[indice:] - (dem[indice]))
        else:
            y = m * x + c
            los = y - (dem[indice:] - (dem[-1]))
        rfresn2 = 0.6 * Modelos.raio_fresnel(1, distancia[indice_d], distancia[-1] - distancia[indice_d], f)
        m2 = -(dem[0] + ht - dem[indice_d] - rfresn2) / distancia[indice_d]
        c2 = (dem[0] + ht - dem[indice_d] - rfresn2)
        x2 = np.array(distancia[:indice_d+1])
        if c2 < 0:
            y2 = m2 * x2
            los2 = y2 - (dem[:indice_d+1] - (dem[0]))
        else:
            y2 = m2 * x2 + c2
            los2 = y2 - (dem[:indice_d+1] - (dem[indice_d]))

        for i in range(len(los) - 1):
            if los[i] < altur_da_cobertuta[i]:
                for n in (0, 1, 2):
                    if landcover[3 * (indice + i) + n] == 10:
                        espesura = espesura + 10  # ( colocar 5, metade dos 10 m)
        altur_da_cobertuta2 = abs(dsm[:indice_d+1] - dem[:indice_d+1])
        for i in range(len(los2) - 2):
            if los2[i] < altur_da_cobertuta2[i]:
                for n in (0, 1, 2):
                    if landcover[3 * i + n] == 10:
                        espesura = espesura + 10  # ( colocar 5, metade dos 10 m)
    """
        if indice - indice_d > 4:
            altur_da_cobertuta3 = dsm[indice_d:indice] - dem[indice_d:indice]
            m3 = -(dem[indice_d] + rfresn2 - dem[indice] - rfresn) / (distancia[indice] - distancia[indice_d])
            c3 = (dem[indice_d] + rfresn2 - dem[indice] - rfresn)
            x3 = np.array(distancia[indice_d:indice]) - distancia[indice_d]

            if c3 < 0:
                y3 = m3 * x3
                los3 = y3 - (dem[indice_d:indice] - (dem[indice_d] + rfresn2))
            else:
                y3 = m3 * x3 + c3
                los3 = y3 - (dem[indice_d:indice] - (dem[indice] + rfresn))
            for i in range(len(los3) - 1):
                if los3[i] < altur_da_cobertuta3[i]:
                    for n in (0, 1, 2):
                        if landcover[3 * (i + indice_d) + n] == 10:
                            espesura = espesura + 10  # ( colocar 5, metade dos 10 m)
    """
    return 0.5*espesura  # considerando 50% da area coberta com vegetação elevada. a documentação dos dados estabelec 10% ou mais/

cobertura = []
markers = [{'lat': 4.9987281, 'lon': 8.3248506, 'nome': 'IME', 'h': 1.7},
           {'lat': -22.9036, 'lon': -43.1895, 'nome': 'PDC', 'h': 4},
           {'lat': 40.0503, 'lon': -105.2600, 'nome': 'MT05P13', 'h': 4}]

#p1 = (markers[1]['lon'], markers[1]['lat'])
#p2 = (markers[0]['lon'], markers[0]['lat'])
#p1 = (markers[2]['lon'], markers[2]['lat'])
f = float(49.72)
ime = 1.7
PDC = 4.2
hg1 = PDC
hg2 = ime


prs=[]
pxs=[]
with open('C:\PythonFlask\PlanCom\medicoes\phase_1\ohpath.txt') as csvfile:
    spamreader = np.genfromtxt(csvfile)
    cont=0

    for row in spamreader:
        if cont!=0:
            ppp = []
            for i in row:
                ppp.append(i)
            prs.append(tuple((ppp[4], ppp[3])))
            pxs.append(tuple((ppp[2], ppp[1])))
        cont += 1
A503V=[]

with open('C:\PythonFlask\PlanCom\medicoes\phase_1\ohdata.txt') as csvfile:
    spamreader = np.genfromtxt(csvfile)
    cont=0

    for row in spamreader:
        if cont!=0:
            ppm = []
            for i in row:
                ppm.append(i)
            A503V.append(ppm[4])
        cont += 1
print(A503V)



perdas=[]
perdas2=[]
perdas3=[]
comparacao=[]

for i in range(len(pxs)):

    dem, dsm, landcover, distancia = perfil(pxs[i], prs[i])
    Densidade_urbana = 0.7
    d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h_urb, visada, indice_visada_r, indice_visada = obter_dados_do_perfil(dem, dsm,
                                                                                                              distancia,                                                                                                          hg1, hg2,
                                                                                                              Densidade_urbana)
    print(d)
    h_urb = h_urb + 0.5
    if (landcover[-1] == 50)or(landcover[-2] == 50)or(landcover[-3] == 50):
        urban = 'wi'
    else:
        urban = 'n'
    yt = 1  # é a perda pelo clima, adotar esse valor padrao inicialmente
    qs = 5  # 70% das situacões
    espesura = obter_vegeta_atravessada(f, indice_visada_r, dem, landcover, dsm, hg2, hg1, distancia, indice_visada)
    # colocar a cidicao para chamar itm ou urbano + espaco livre

    h0 = (dem[0] + dem[-1]) / 2
    demsm=dsm
    if (indice_visada > 1) and indice_visada_r != indice_visada:
        for u in range(indice_visada-1):
            demsm[u]=dem[u]
        for u in range(indice_visada_r, len(demsm)):
            demsm[u]= dem[u]

    else:
        demsm=dem
    dls, hs = parametros_difracao(distancia, dem, hg1, hg2, f)

    epstein = Modelos.modelo_epstein_peterson(dls, hs, f)
    espaco_livre = Modelos.friis_free_space_loss_db(f, d)
    itm, variabilidade_situacao, At, dls_LR = Modelos.longLq_rice_model(h0, f, hg1, hg2, he1, he2, d, yt, qs, dl1, dl2, Dh, visada,
                                      teta1, teta2, polarizacao='v', simplificado=0)

    if urban == 'wi' and h_urb > hg2 + 0.5:
        urb = max(0, Modelos.ikegami_model(h_urb, hg2, f))
    else:
        urb = 0
    vegetacao = Modelos.atenuaca_vegetacao_antiga_ITU(f, espesura)
    rearth = Modelos.opcional_ar(f,h0,d, he1, he2)
    terreno = max(rearth, epstein)
    total_itm = espaco_livre+ urb + vegetacao + itm + variabilidade_situacao
    total_epstein_peterson = espaco_livre+ urb + vegetacao + epstein
    comparacao.append((epstein,  itm,vegetacao,urb, A503V[i], Dh))
    perdas.append(itm+vegetacao+urb+variabilidade_situacao)
    perdas2.append(epstein+vegetacao+urb)

    if (Dh>90) and (d<=0.7*dls_LR):
        pd3=epstein + vegetacao + urb
        perdas3.append(pd3)
    else:
        pd3=itm+vegetacao+urb+variabilidade_situacao
        perdas3.append(pd3)

    with open("ohteste.txt", "a") as arquivo:
        arquivo.write("\n"+str(pxs[i][0])+","+str(pxs[i][1])+","+str(prs[i][0])+","+str(prs[i][1])+","+str(d)+","+str(epstein)+","+str(itm+variabilidade_situacao)+","+str(vegetacao)+","+str(urb)+","+str(epstein+vegetacao+urb)+","+str(itm+vegetacao+urb+variabilidade_situacao)+","+str(pd3)+","+str(A503V[i]))

perdas = np.array(perdas)
diferenca = []
diferenca2 = []
diferenca3 = []

for i in range(len(A503V)):
    if A503V[i] < 100:
        diferenca.append(A503V[i]-perdas[i])
        diferenca2.append(A503V[i] - perdas2[i])
        diferenca3.append(A503V[i] - perdas3[i])

diferenca = np.array(diferenca)
diferenca2 = np.array(diferenca2)
diferenca3 = np.array(diferenca3)

med=np.mean(diferenca)
medquadrati=np.mean(diferenca**2)
med2=np.mean(diferenca2)
medquadrati2=np.mean(diferenca2**2)
med3=np.mean(diferenca3)
medquadrati3=np.mean(diferenca3**2)

print(med)
print(medquadrati)
print(med2)
print(medquadrati2)
print(med3)
print(medquadrati3)

print(comparacao)

"""dem, dsm, landcover, distancia = perfil(p1, p2)
Densidade_urbana = 0.7
d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h_urb, visada, indice_visada_r, indice_visada = obter_dados_do_perfil(
    dem, dsm,
    distancia, hg1, hg2,
    Densidade_urbana)
if landcover[-1] == 50:
    urban = 'wi'
else:
    urban = 'n'
yt = 1  # é a perda pelo clima, adotar esse valor padrao inicialmente
qs = 7  # 70% das situacões
espesura = obter_vegeta_atravessada(f, indice_visada_r, dem, landcover, dsm, hg2, hg1, distancia, indice_visada)
# colocar a cidicao para chamar itm ou urbano + espaco livre

h0 = (dem[0] + dem[-1]) / 2

dls, hs = parametros_difracao(distancia, dem, hg1, hg2)

epstein = Modelos.modelo_epstein_peterson(dls, hs, f)
espaco_livre = Modelos.friis_free_space_loss_db(f, d)
itm, variabilidade_situacao, At = Modelos.longLq_rice_model(h0, f, hg1, hg2, he1, he2, d, yt, qs, dl1, dl2, Dh, visada,
                                                            teta1, teta2, polarizacao='v', simplificado=0)

if urban == 'wi' and h_urb > hg2 + 0.5:
    urb = Modelos.ikegami_model(h_urb, hg2, f)
else:
    urb = 0
vegetacao = Modelos.atenuaca_vegetacao_antiga_ITU(f, espesura)
total_itm = espaco_livre + urb + vegetacao + itm + variabilidade_situacao
total_epstein_peterson = espaco_livre + urb + vegetacao + epstein
perdas.append((espaco_livre, urb, vegetacao, itm, variabilidade_situacao, epstein, total_itm, total_epstein_peterson))

print(perdas)
print(
    f' ({f}, {hg1}, {hg2}, {he1}, {he2}, {d}, {yt}, {qs}, {dl1}, {dl2}, {Dh}, {visada},{teta1}, {teta2}, {urban})')

plt.plot(distancia, dem)
plt.title('Modelo Digital de Elevação (DEM)')
plt.show()
"""
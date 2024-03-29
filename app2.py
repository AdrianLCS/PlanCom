import numpy as np
import rasterio
from flask import Flask, render_template, request, jsonify
import os
import folium
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
app = Flask(__name__)
c = 299792458  # m/s
a = 6378137  # m
b = 6356752  # m

Configuracao = {"modelo": "ITM", "urb": 1, "veg": 1, "precisao": 0.5, "max_alt": 300, "min_alt": 0}  # ITM ou Epstein-peterson
mapas = []

def extrair_vet_area(raio, ponto, f, limear, unidade_distancia, precisao):
    comprimento_de_onda = c / (f * 1000000)
    # L_db = -20 * np.log10(comprimento_de_onda) + 20 * np.log10(d) + 22
    d = 10 ** ((limear / 20) + np.log10(comprimento_de_onda) - 1.1)
    d = min(raio, d)
    print(d)
    retas = []
    dem0, dsm0, landcover0, distancia0 = [], [], [], []
    for i in range(int(360*precisao)):
        vet = np.array([np.cos(i * 2 * np.pi / (precisao*360)), np.sin(i * 2 * np.pi /(precisao*360))]) # roda no sentido positivo trigonométrio de 0.5 em 0.5 graus
        pf=np.array(ponto) + vet*(
                d / unidade_distancia)*(1/3600)
        print(ponto)
        print(pf)
        r = reta(ponto, pf)
        dem, dsm, landcover, distancia = perfil(r,1)
        distancia0.append(distancia)
        retas.append(r)
        dem0.append(dem)
        dsm0.append(dsm)
        landcover0.append(landcover)
    print('criou as retas')
    return retas, d, dem0, dsm0, landcover0, distancia0


def parametros_difracao(distancia, dem, ht, hr):
    angulo = []
    demr = dem[::-1]
    d = distancia[-1]
    aref = np.arctan((-ht - dem[0] + hr + demr[0]) / d)
    visada = 1  # 'visada# '
    maxangulo = aref
    idl1 = 0
    dls = [0]
    hs = [ht + dem[0]]
    h, dl1, teta1 = 0, 0, 0
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
        maxangulo = aref
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
    hs.append(dem[-1] + hr)
    return dls, hs


def calcula_perda(ht, hr, f, r):
    dem, dsm, landcover, distancia = perfil(r)
    Densidade_urbana = 0.7
    d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h, visada, indice_visada_r, indice_visada = obter_dados_do_perfil(
        dem, dsm, distancia, ht, hr, Densidade_urbana)

    if landcover[-1] == 50:
        urb = Modelos.ikegami_model(h, hg2, f)
    else:
        urb = 0
    espaco_livre = Modelos.friis_free_space_loss_db(f, d)

    yt = 1  # é a perda pelo clima, adotar esse valor padrao inicialmente
    qs = 7  # 70% das situacões
    espesura, h = obter_vegeta_atravessada(f, indice_visada_r, dem, landcover, dsm, hr, ht, distancia, indice_visada)

    # colocar a cidicao para chamar itm ou urbano + espaco livre
    vegetacao = Modelos.atenuaca_vegetacao_antiga_ITU(f, espesura)
    # print(
    #    f' ({f}, {hg1}, {hg2}, {he1}, {he2}, {d}, {yt}, {qs}, {dl1}, {dl2}, {Dh}, {visada}, {h},{teta1}, {teta2}, {d_urb}, {hb_urb}, {urban})')
    hmed = (dem[0] + dem[-1]) / 2
    perda, variabilidade_situacao, At = Modelos.longLq_rice_model(hmed, f, hg1, hg2, he1, he2, d, yt, qs, dl1, dl2,
                                                                  Dh, visada,
                                                                  teta1, teta2, polarizacao='v')

    return perda


def modificar_e_salvar_raster(raster_path, ponto, raio, limear, ht, hr, f, precisao):
    pasta = raster_path[:-11] + 'modificado'
    file = '\A' + raster_path[-11:]
    yt = 1
    qs = 7
    unidade_distancia = 2 * np.pi * R(ponto[1]) / (1296000)
    retas, raio, dem0, dsm0, landcover0, distancia0 = extrair_vet_area(raio, ponto, f, limear, unidade_distancia, precisao)
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
                    angulo2 = int((180 * angulo / np.pi)*precisao)
                    r = retas[angulo2][:int(distyx+1)]
                    dem = dem0[angulo2][:int(distyx+1)]
                    dsm = dsm0[angulo2][:int(distyx+1)]
                    landcover = landcover0[angulo2][:3 * int(distyx) + 1]
                    distancia = distancia0[angulo2][:int(distyx+1)]

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
                        perda, variabilidade_situacao, At = Modelos.longLq_rice_model(hmed, f, hg1, hg2, he1, he2, d,
                                                                                      yt, qs, dl1,
                                                                                      dl2,
                                                                                      Dh, visada,
                                                                                      teta1, teta2, polarizacao='v')

                        espaco_livre = Modelos.friis_free_space_loss_db(f, d)
                        espesura = obter_vegeta_atravessada(f, indice_visada_r, dem, landcover, dsm, hr, ht, distancia, indice_visada)
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


def R(lat):
    return (((((a ** 2) * np.cos(lat * np.pi / 180)) ** 2) + (((b ** 2) * np.sin(lat * np.pi / 180)) ** 2)) / (
            ((a * np.cos(lat * np.pi / 180)) ** 2) + ((b * np.sin(lat * np.pi / 180)) ** 2))) ** 0.5


def obter_dados_do_raster(indice_atual, r, dem, dsm, landcover, d, distancia, area):
    caminho, caminho_dsm, caminho_landcover = obter_raster(r[indice_atual], r[indice_atual])
    with rasterio.open(caminho) as src, rasterio.open(caminho_dsm) as src_dsm, rasterio.open(
            caminho_landcover) as src_landcover:
        raster = src.read(1)
        raster_dsm = src_dsm.read(1)
        raster_landcover = src_landcover.read(1)
        inv_transform = ~src.transform
        inv_transform_dsm = ~src_dsm.transform
        inv_transform_landcover = ~src_landcover.transform


        for i in range(np.shape(r)[0]):
            if (np.floor(r[i][0]) == np.floor(r[indice_atual][0])) and (np.floor(r[i][1]) == np.floor(r[indice_atual][1])):
                pixel_x1, pixel_y1 = inv_transform * (r[i][0], r[i][1])
                pixel_x1_dsm, pixel_y1_dsm = inv_transform_dsm * (r[i][0], r[i][1])
                pixel_x1_lancover, pixel_y1_landcover = inv_transform_landcover * (r[i][0], r[i][1])
                dist = distancia * i
                if area:
                    alt_dem = raster[int(pixel_y1)][int(pixel_x1)]
                    alt_dsm = raster_dsm[int(pixel_y1_dsm)][int(pixel_x1_dsm)]
                else:
                    alt_dem = ((1 - (pixel_y1 - np.floor(pixel_y1))) * raster[int(np.floor(pixel_y1))][
                        int(np.floor(pixel_x1))] + (
                                       pixel_y1 - np.floor(pixel_y1)) * raster[int(np.ceil(pixel_y1))][
                                   int(np.floor(pixel_x1))] + (
                                       1 - (pixel_x1 - np.floor(pixel_x1))) * raster[int(np.floor(pixel_y1))][
                                   int(np.floor(pixel_x1))] + (
                                       pixel_x1 - np.floor(pixel_x1)) * raster[int(np.floor(pixel_y1))][
                                   int(np.ceil(pixel_x1))]) / 2

                    alt_dsm = ((1 - (pixel_y1_dsm - np.floor(pixel_y1_dsm))) * raster_dsm[int(np.floor(pixel_y1_dsm))][
                        int(np.floor(pixel_x1_dsm))] + (
                                       pixel_y1 - np.floor(pixel_y1_dsm)) * raster_dsm[int(np.ceil(pixel_y1_dsm))][
                                   int(np.floor(pixel_x1_dsm))] + (
                                       1 - (pixel_x1_dsm - np.floor(pixel_x1_dsm))) *
                               raster_dsm[int(np.floor(pixel_y1_dsm))][
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
                    pixel_x2_lancover, pixel_y2_landcover = inv_transform_landcover * (
                        r[i][0] + lonpasso, r[i][1] + latpasso)
                    pixel_x3_lancover, pixel_y3_landcover = inv_transform_landcover * (
                        r[i][0] + 2 * lonpasso, r[i][1] + 2 * latpasso)
                    landcover.append(raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)])
                    landcover.append(raster_landcover[int(pixel_y3_landcover)][int(pixel_x3_lancover)])
                indice_atual = i
            else:
                indice_atual = i
                break
    return dem, dsm, landcover, d, indice_atual


def perfil(r, area=0):
    unidade_distancia = 2 * np.pi * R(r[0][1]) / (1296000)
    indice_atual = 0
    dem = []
    dsm = []
    landcover = []
    d = []
    caminho, caminho_dsm, caminho_landcover = obter_raster(r[0], r[0])
    with rasterio.open(caminho) as src:
        inv_transform = ~src.transform
        pixel_xn, pixel_yn = inv_transform * (r[np.shape(r)[0] - 1][0], r[np.shape(r)[0] - 1][1])
        x0, y0 = inv_transform * (r[0][0], r[0][1])
        distancia = unidade_distancia * ((((pixel_xn - x0) ** 2) + ((pixel_yn - y0) ** 2)) ** 0.5) / (
                np.shape(r)[0] - 1)

    while indice_atual < np.shape(r)[0]-1:
        dem, dsm, landcover, d, indice_atual = obter_dados_do_raster(indice_atual, r, dem, dsm, landcover, d, distancia, area)

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
    dl1, dl2, teta1, teta2 = d, d, None, None
    maxangulo = aref
    maxangulor = -aref
    indice_visada=0

    for i in range(1, len(dem) - 1):
        angulo.append(np.arctan((dem[i] - (dem[0] + ht)) / distancia[i]))
        if (angulo[i - 1] > aref) and (angulo[i - 1] > maxangulo):
            teta1, dl1, idl1 = angulo[i - 1], distancia[i], i
            indice_visada=idl1
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
    h_urb = max(0, (1 / Densidade_urbana) * np.mean(dsm[-3:len(dsm)]) - np.mean(dem[-3:len(dem)]))

    return d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h_urb, visada, indice_visada_r, indice_visada


def obter_vegeta_atravessada(f, indice, dem, landcover, dsm, hr, ht, distancia, indice_d):
    dem = np.array(dem)
    dsm = np.array(dsm)

    altur_da_cobertuta = dsm[indice:] - dem[indice:]
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
        rfresn = 0.6 * Modelos.raio_fresnel(1, distancia[indice], distancia[-1] - distancia[indice], f)
        m = -(rfresn + dem[indice] - dem[-1] - hr) / ((len(dem) - (indice + 1)) * distancia[1])
        c = (rfresn + dem[indice] - dem[-1] - hr)
        x = np.array(distancia[indice:])
        x = x - distancia[indice]
        if c < 0:
            y = m * x
            los = y - (dem[indice:] - (dem[indice] + rfresn))
        else:
            y = m * x + c
            los = y - (dem[indice:] - (dem[-1] + hr))
        rfresn2 = 0.6 * Modelos.raio_fresnel(1, distancia[indice_d], distancia[indice_d], f)
        m2 = -(dem[0] + ht - dem[indice_d] - rfresn2) / distancia[indice_d]
        c2 = (dem[0] + ht - dem[indice_d] - rfresn2)
        x2 = np.array(distancia[:indice_d])
        if c2 < 0:
            y2 = m2 * x2
            los2 = y2 - (dem[:indice_d] - (dem[0] + ht))
        else:
            y2 = m2 * x2 + c2
            los2 = y2 - (dem[:indice_d] - (dem[indice_d] + rfresn2))

        for i in range(len(los) - 1):
            if los[i] < altur_da_cobertuta[i]:
                for n in (0, 1, 2):
                    if landcover[3 * (indice_d + i) + n] == 10:
                        espesura = espesura + 10  # ( colocar 5, metade dos 10 m)
        altur_da_cobertuta2 = dsm[:indice_d] - dem[:indice_d]
        print(espesura)
        for i in range(len(los2) - 2):
            if los2[i] < altur_da_cobertuta2[i]:
                for n in (0, 1, 2):
                    if landcover[3 * i + n] == 10:
                        espesura = espesura + 10  # ( colocar 5, metade dos 10 m)
        print(espesura)
    return espesura





cobertura = [{'nome': 'PDC_Area_de_cobertura_800Mhz', 'raster': 'raster\S23W044.tif','f': 800, 'img': 'Raster\modificado\AS23W044.png', 'h': 10}]
#cobertura=[]
def addfoliun():
    global Configuracao
    escala_de_altura = [Configuracao["min_alt"], Configuracao["max_alt"]]

    # ['00FFFF','00FFCC','33CCCC','669999','996699', 'CC3366', 'FF3366','FF0033','FF0000']
    image_viz_params = {'bands': ['elevation'], 'min': escala_de_altura[0], 'max': escala_de_altura[1],
                        'palette': ['0000ff', '00ffff', 'ffff00', 'ff0000', 'ffffff'],
                        'opacity': 0.5}  # Altura limitada par visualização BR

    # map_elevn = geemap.Map(center=[-22.9120, -43.2089], zoom=10)
    # map_elevn.add_layer(elvn, image_viz_params, 'Elevacao')

    folium_map = folium.Map(location=[-22.9120, -43.2089], zoom_start=7)
    try:

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

        folium.TileLayer(tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
                         attr='OpenTopoMap',
                         name='OpenTopoMap').add_to(folium_map)
    except:
        print("erro ao tentar acessar a internet")
    try:
        ee.Authenticate()
        ee.Initialize(project="plancom-409417")
        elvn = ee.Image("NASA/NASADEM_HGT/001")
        map_id_dict = ee.Image(elvn).getMapId(image_viz_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
            name='elevacao',
            overlay=True,
            control=True
        ).add_to(folium_map)
    except:
        print('Erro na biblioteca Earth Engine')
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
           {'lat': -22.9036, 'lon': -43.1895, 'nome': 'PDC', 'h': 22.5}]


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

            r = reta(p1, p2)
            f = float(request.form.get("f"))
            dem, dsm, landcover, distancia = perfil(r)
            Densidade_urbana = 0.7
            d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h_urb, visada, indice_visada_r, indice_visada = obter_dados_do_perfil(
                dem, dsm, distancia, ht, hr, Densidade_urbana)
            if landcover[-1] == 50:
                urban = 'wi'
            else:
                urban = 'n'
            yt = 1  # é a perda pelo clima, adotar esse valor padrao inicialmente
            qs = 7  # 70% das situacões

            espesura = obter_vegeta_atravessada(f, indice_visada_r, dem, landcover, dsm, hr, ht, distancia, indice_visada)
            # colocar a cidicao para chamar itm ou urbano + espaco livre

            print(
                f' ({f}, {hg1}, {hg2}, {he1}, {he2}, {d}, {yt}, {qs}, {dl1}, {dl2}, {Dh}, {visada}, {h_urb},{teta1}, {teta2}, {urban})')
            hmed = (dem[0] + dem[-1]) / 2
            perda, variabilidade_situacao, At = Modelos.longLq_rice_model(hmed, f, hg1, hg2, he1, he2, d, yt, qs, dl1,
                                                                          dl2,
                                                                          Dh, visada,
                                                                          teta1, teta2, polarizacao='v')

            espaco_livre = Modelos.friis_free_space_loss_db(f, d)

            if urban == 'wi' and h_urb > hg2 + 0.5:
                urb = Modelos.ikegami_model(h_urb, hg2, f)
            else:
                urb = 0

            vegetacao = Modelos.atenuaca_vegetacao_antiga_ITU(f, espesura)

            # colocar aqu uma funcao que adiciona a perda por vegetacao
            print(
                f' Perda por popgação no espaço livre: {espaco_livre}\n Perda relativa ao terreno: {perda}\n perda por variabilidade da situação: {variabilidade_situacao} \n perda em vegetação: {vegetacao} \b Perda urbana: {urb}')
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
        hr = 2
        caminho, caminho_dsm, caminho_landcover = obter_raster(p1, p1)
        precisao=0.5
        caminho = modificar_e_salvar_raster(caminho, p1, float(request.form.get("raio")), limear, ht, hr, float(request.form.get("f")), precisao)

        img = criaimg(caminho)
        cobertura.append({'nome': request.form.get("ponto") + '_Area_de_cobertura' + '_' + request.form.get("f"), 'raster': caminho, 'f': float(request.form.get("f")), 'img': img, 'h': ht})
    return render_template('area.html')


@app.route('/config', methods=['GET', 'POST'])
def conf():
    global Configuracao
    p1 = ()
    ht = 2
    if request.form.get("modelo") and request.form.get("urb") and request.form.get("veg") and \
            request.form.get("precisao") and request.form.get("min_alt") and request.form.get("max_alt"):
        Configuracao = {"modelo": request.form.get("modelo"),
                        "urb": request.form.get("urb"),
                        "veg": request.form.get("veg"),
                        "precisao": request.form.get("precisao"),
                        "max_alt": request.form.get("max_alt"),
                        "min_alt": request.form.get("min_alt")}  # ITM ou Epstein-peterson
    return render_template('conf.html')


@app.route('/addmapa', methods=['GET', 'POST'])
def addmapa():
    global mapas
    if request.form.get("mapa") :
        mapas.append(request.form.get("mapa"))
    return render_template('addmapa.html')


if __name__ == '__main__':
    app.run(debug=True)

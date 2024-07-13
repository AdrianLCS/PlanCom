import numpy as np
import rasterio
from flask import Flask, render_template, redirect, url_for, request, session, jsonify, flash
import os
import folium
import pickle
import matplotlib.pyplot as plt
import Modelos
from PIL import Image

# Váriaves Globais
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'tif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Chave secreta para criptografar sessões
users = {'adrian': 'adrian', 'user2': 'password2'}
c = 299792458  # m/s
a = 6378137  # m
b = 6356752  # m
Configuracao = {"urb": 1, "veg": 1, "precisao": 4, "largura_da_rua": 22.5, "alt_max": 100}
# Criar opção de adiconar rádio #sensibilidade em e potencia W ganho em dB frequencia em MHz
radio1 = {'nome': '"rf7800v"', 'sensibilidade': -116, 'faixa_de_freq': [30, 108],
          'potencia': {'tipo': 1, 'valor': [0.25, 1, 2, 5, 10]},
          'antenas': [{'nome': 'wip', 'tiopo': 0, 'ganho': 0}, {'nome': 'bade', 'tiopo': 0, 'ganho': 1}]}
radio2 = {'nome': 'APX2000', 'sensibilidade': -102, 'faixa_de_freq': [806, 870],
          'potencia': {'tipo': 0, 'valor': [1, 3]},
          'antenas': [{'nome': 'wip', 'tiopo': 0, 'ganho': 3}]}
radios = [radio1, radio2]


def generate_raster_files(base_string):
    """Obtém uma lista de arquivos Raster com os Rasters em torno de um nome de Raster de entrada. Usada na função
    unir_raster_3x3 """
    import re

    # Extrair a coordenada base da string
    match = re.search(r'([NS])(\d{2})([EW])(\d{3})', base_string)
    if not match:
        raise ValueError("String de entrada não está no formato esperado.")

    base_lat_dir = match.group(1)
    base_lat = int(match.group(2))
    base_lon_dir = match.group(3)
    base_lon = int(match.group(4))

    # Converter direções N/S e E/W para 1 ou -1
    lat_sign = 1 if base_lat_dir == 'N' else -1
    lon_sign = 1 if base_lon_dir == 'E' else -1

    # Função auxiliar para gerar a direção correta
    def get_lat_dir(lat):
        return 'N' if lat >= 0 else 'S'

    def get_lon_dir(lon):
        return 'E' if lon >= 0 else 'W'

    # Gerar a lista de arquivos raster na ordem especificada
    raster_files = []
    for lat_offset in range(-1, 2):
        for lon_offset in range(-1, 2):
            lat = base_lat + lat_offset
            lon = base_lon + lon_offset
            lat_dir = get_lat_dir(lat_sign * lat)
            lon_dir = get_lon_dir(lon_sign * lon)
            lat = abs(lat)
            lon = abs(lon)
            raster_files.append(f"Raster\\{lat_dir}{lat:02d}{lon_dir}{lon:03d}.tif")

    return raster_files


def unir_raster_3x3(raster_path):
    """Uir 9 Rasters, usada para formar um raster maior para gerar a área de cobertura quando essa abrange mais de um
    Raster """
    raster_files = raster_files = generate_raster_files(raster_path)

    # Nome do arquivo de saída
    output_raster = raster_path[:-4] + "_3x3.tif"

    # Número de rasters ao longo de uma linha ou coluna
    num_rasters_row_col = 3

    # Abrir os nove rasters

    with rasterio.open(raster_files[0]) as src0, rasterio.open(raster_files[1]) as src1, rasterio.open(
            raster_files[2]) as src2, \
            rasterio.open(raster_files[3]) as src3, rasterio.open(raster_files[4]) as src4, rasterio.open(
        raster_files[5]) as src5, \
            rasterio.open(raster_files[6]) as src6, rasterio.open(raster_files[7]) as src7, rasterio.open(
        raster_files[8]) as src8:
        srcs = []
        srcs.append(src0), srcs.append(src1), srcs.append(src2), srcs.append(src3), srcs.append(src4)
        srcs.append(src5), srcs.append(src6), srcs.append(src7), srcs.append(src8)
        # Obter informações sobre um dos rasters (usaremos o primeiro como referência)
        profile = srcs[0].profile
        transform = srcs[0].transform

        # Criar um array 2D para armazenar os dados combinados
        combined_data = np.zeros((srcs[0].height * num_rasters_row_col, srcs[0].width * num_rasters_row_col),
                                 dtype=np.float32)

        # Inserir os dados de cada raster na posição apropriada no array combinado
        for i in range(num_rasters_row_col):
            for j in range(num_rasters_row_col):
                src = srcs[i * num_rasters_row_col + j]
                data = src.read(1)
                combined_data[i * src.height:(i + 1) * src.height, j * src.width:(j + 1) * src.width] = data

    # Atualizar as informações do perfil do raster
    profile.update(count=1, width=combined_data.shape[1], height=combined_data.shape[0], transform=transform)

    # Salvar o raster combinado
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(combined_data, 1)
    return output_raster


def deg2rad(degrees):
    """Converte graus para radianos"""
    radians = degrees * np.pi / 180
    return radians


def getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2):
    """Formula para cálculo de distância entre duas coordenadas"""
    theta = longitude1 - longitude2
    latitude1 = deg2rad(latitude1)
    latitude2 = deg2rad(latitude2)
    longitude1 = deg2rad(longitude1)
    longitude2 = deg2rad(longitude2)
    distance = 2 * R((latitude1 + latitude2) / 2) * np.arcsin(((np.sin((latitude2 - latitude1) / 2)) ** 2 +
                                                               np.cos(latitude1) * np.cos(latitude2) * ((np.sin(
                (longitude2 - longitude1) / 2)) ** 2)) ** 0.5)

    return distance


def allowed_file(filename):
    """Verifica se o arquivo carregado é compatível"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extrair_vet_area(raio, ponto, f, limear, unidade_distancia, precisao,  local_Configuracao):
    """Gera os perfis em intervalos azimutais dado pela Váriavel Configuracao, para o cáculo de área de cobertura"""
    comprimento_de_onda = c / (f * 1000000)
    # L_db = -20 * np.log10(comprimento_de_onda) + 20 * np.log10(d) + 22
    d = 10 ** ((limear / 20) + np.log10(comprimento_de_onda) - 1.1)
    d = min(raio, d)
    qtd_pontos = int(np.ceil(d / unidade_distancia))
    qtd_retas = int(360 * precisao)
    retas = []
    # dem0, dsm0, landcover0, distancia0 = [], [], [], []
    dem0, dsm0, landcover0, distancia0 = np.zeros((qtd_retas, qtd_pontos)), np.zeros((qtd_retas, qtd_pontos)), np.zeros(
        (qtd_retas, 3 * (qtd_pontos - 1) + 1)), np.zeros((qtd_retas, qtd_pontos))

    for i in range(int(360 * precisao)):
        vet = np.array([np.cos(i * 2 * np.pi / qtd_retas), np.sin(
            i * 2 * np.pi / qtd_retas)])  # roda no sentido positivo trigonométrio de 2 em 2 graus
        pf = np.array(ponto) + vet * (
                d / unidade_distancia) * (1 / 3600)
        dem, dsm, landcover, distancia, r = perfil(ponto, pf,  local_Configuracao, 1)
        # distancia0.append(distancia)
        distancia0[i] = distancia
        retas.append(r)
        # dem0.append(dem)
        dem0[i] = dem
        # dsm0.append(dsm)
        dsm0[i] = dsm
        # landcover0.append(landcover)
        landcover0[i] = landcover
        print(i)
    return retas, d, dem0, dsm0, landcover0, distancia0


def parametros_difracao(distancia, dem, ht, hr):
    angulo = []
    d = distancia[-1]
    aref = (hr + dem[-1] - ht - dem[0]) / d
    visada = 1  # 'visada# '
    maxangulo = aref
    dls = [0]
    hs = [ht + dem[0]]
    h, idl1, teta1 = 0, 0, 0
    for i in range(1, len(dem) - 1):
        angulo.append((dem[i] - (dem[0] + ht)) / distancia[i])
        if angulo[-1] > maxangulo:
            idl1 = i
            h = dem[i]
            visada = 0
            maxangulo = max(angulo)
    if not visada:
        hs.append(h)
        dls.append(distancia[idl1])

    while not visada:
        idll = [idl1]
        angulo = []
        aref = (hr + dem[-1] - dem[idl1]) / (d - distancia[idl1])
        maxangulo = aref
        visada = 1
        for i in range(idl1 + 3, len(dem) - 1):
            angulo.append((dem[i] - (dem[idl1])) / (distancia[i] - distancia[idl1]))
            if (angulo[-1] > maxangulo):
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


def modificar_e_salvar_raster(raster_path, ponto, raio, limear, ht, hr, f, precisao, largura_da_rua, local_Configuracao):
    """Essa é a principal função para gerar uma área de cobertura ela modifica um raster de DEM substituindo os
    valores por dois valores padronizados um para quando o enlace é possível e outro para quano o enlace não é possível.
    Uma imagem será gerada a partir desse raster com a função criaimg"""
    pasta = raster_path[:-11] + 'modificado'
    file = '\A' + raster_path[-11:]
    yt = 1
    qs = 5

    with rasterio.open(raster_path, 'r+') as src:
        # Ler a matriz de dados do raster
        data = src.read(1)
        inv_transform = ~src.transform
        transform = src.transform
        x, y = inv_transform * (ponto[0], ponto[1])

    unidade_distancia = 2 * np.pi * R(ponto[1]) / (360 * (1 / transform[0]))
    retas, raio, dem0, dsm0, landcover0, distancia0 = extrair_vet_area(raio, ponto, f, limear, unidade_distancia,
                                                                       precisao,  local_Configuracao)
    xy = min(x, 3600 - x, y, 3600 - y)
    if xy * unidade_distancia <= raio:
        raster_unido = unir_raster_3x3(raster_path)
        raster_path = raster_unido

    with rasterio.open(raster_path, 'r+') as src:
        # Ler a matriz de dados do raster
        data = src.read(1)
        inv_transform = ~src.transform
        transform = src.transform
        x, y = inv_transform * (ponto[0], ponto[1])

        # Abrir o arquivo raster para leitura e escrita

        # Modificar o valor do ponto desejado
        print('retas obtidas')
        print('percorrendo raster')
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
                    r = retas[angulo2][:int(distyx)]
                    dem = dem0[angulo2][:int(distyx)]
                    dsm = dsm0[angulo2][:int(distyx)]
                    landcover = landcover0[angulo2][:3 * int(distyx - 1) + 1]
                    distancia = distancia0[angulo2][:int(distyx)]
                    Densidade_urbana = 1
                    d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h_urb, visada, indice_visada_r, indice_visada = obter_dados_do_perfil(
                        dem, dsm, distancia, ht, hr, Densidade_urbana)

                    min_alt = Modelos.min_alt_ikegami(f)
                    if h_urb > float(local_Configuracao["alt_max"]):
                        h_urb = float(local_Configuracao["alt_max"]) + min_alt
                    else:
                        h_urb = h_urb + min_alt

                    hmed = (dem[0] + dem[-1]) / 2

                    if visada:
                        data[linha][coluna] = 2
                    else:

                        if local_Configuracao["urb"]:
                            if ((landcover[-1] == 50)or(landcover[-2] == 50) or(landcover[-3] == 50)) and (
                                    h_urb > hg2 + min_alt):
                                urb = Modelos.ikegami_model(h_urb, hg2, f, w=float(largura_da_rua))
                            else:
                                urb = 0
                        else:
                            urb = 0

                        if local_Configuracao["veg"]:
                            espesura = obter_vegeta_atravessada(f, indice_visada_r, dem, landcover, dsm, hr, ht,
                                                                distancia,
                                                                indice_visada)
                            vegetacao = Modelos.atenuaca_vegetacao_antiga_ITU(f, espesura)
                        else:
                            vegetacao = 0

                        dls, hs = parametros_difracao(distancia, dem, hg1, hg2)
                        espaco_livre = Modelos.friis_free_space_loss_db(f, d)
                        epstein = Modelos.modelo_epstein_peterson(dls, hs, f)
                        itm, variabilidade_situacao, At, dls_LR = Modelos.longLq_rice_model(hmed, f, hg1, hg2, he1,
                                                                                            he2,
                                                                                            d,
                                                                                            yt, qs, dl1,
                                                                                            dl2,
                                                                                            Dh, visada,
                                                                                            teta1, teta2,
                                                                                            polarizacao='v')

                        if (Dh > 100) and (d <= 0.7 * dls_LR) or (d < 0.1 * dls_LR):
                            p = (espaco_livre + epstein + vegetacao + urb)
                        else:
                            p = (espaco_livre + itm + vegetacao + urb + variabilidade_situacao)

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


def criaimg(dem_file, nova_cobertura):
    """Essa função é usada para criar uma imagem a partir de um Raster de canal único, é usada para formar a imagem
    que será visualizada como área de cobertura. O raster usado na entrada da função é gerado pela função
    modificar_e_salvar_raster """
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
    plt.savefig('Raster/modificado/' + nova_cobertura + ".png", format="png", bbox_inches='tight', pad_inches=0)

    # Fechar a figura para liberar recursos
    plt.close()
    return 'Raster/modificado/' + nova_cobertura + ".png"


def carregamapa(caminho_completo, filename):
    """Essa função forma a camada de visualização de uma carta carregada pelo usuário sobre o mapa em função de uma
    imagem e um raster para referências das coordenadas geográficas """
    # Carregar o arquivo DEM (tif)
    dem_dataset = rasterio.open(caminho_completo)
    # Obter as informações sobre a extensão do DEM
    bounds = dem_dataset.bounds
    min_lat, min_lon = bounds.bottom, bounds.left
    max_lat, max_lon = bounds.top, bounds.right
    nome = filename

    # Adicionar a camada do raster como uma sobreposição de imagem
    image_overlay = folium.raster_layers.ImageOverlay(
        image=caminho_completo[:-4] + '.jpg',
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=1,
        name=nome,
        interactive=True,
        cross_origin=False,
        zindex=1
    )
    return image_overlay


def criamapa(dem_file, img_file, local_cobertura):
    """Essa função forma a camada de visualização da área de cobertura sobre o mapa em função de uma imagem e um
    raster para referências das coordenadas geográficas """
    # Carregar o arquivo DEM (tif)
    dem_dataset = rasterio.open(dem_file)

    # Obter as informações sobre a extensão do DEM
    bounds = dem_dataset.bounds
    min_lat, min_lon = bounds.bottom, bounds.left
    max_lat, max_lon = bounds.top, bounds.right

    # Calcular o centro do DEM para definir o local inicial do mapa
    nome = '0'
    for i in local_cobertura:
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
    """Essa funão forama uma Array contendo todas as coordenadas (longitude, latitude) em graus decimais entre
    empaçadas aproximadamente 1 em 1 segundo de arco entre numa reta entre dois pontos """
    p1 = np.array(p1)
    p2 = np.array(p2)
    v = p2 - p1
    modulo = np.linalg.norm(v)
    n = int(np.ceil(modulo / tranform))  # tranform=1/3600
    t = np.linspace(0, 1, n)
    r = []
    for i in t:
        r.append(p1 + v * i)
    r = np.array(r)
    dist = getDistanceBetweenPointsNew(p1[1], p1[0], p2[1], p2[0]) / (n - 1)
    return r, dist


def R(lat):
    """Essa função retorna o raio da terra em função da latitude. Usada para o cáculuo de distância entre dois pontos"""
    return (((((a ** 2) * np.cos(lat * np.pi / 180)) ** 2) + (((b ** 2) * np.sin(lat * np.pi / 180)) ** 2)) / (
            ((a * np.cos(lat * np.pi / 180)) ** 2) + ((b * np.sin(lat * np.pi / 180)) ** 2))) ** 0.5


def obter_dados_do_raster(indice_atual, r, dem, dsm, landcover, d, distancia, area, local_Configuracao):
    """Essa função extrai o perfil de elevação superfífice e land Cover ao longo do caminho entre dois pontos dentro
    de um mesmo arquivo Raster. """
    caminho, caminho_dsm, caminho_landcover = obter_raster(r[indice_atual], r[indice_atual])
    if ( local_Configuracao["urb"] or local_Configuracao["veg"]) or not area:
        with rasterio.open(caminho) as src:
            raster = src.read(1)
            inv_transform = ~src.transform
            # transform = src.transform
            # londem0, latdem0 = transform * (0, 0)
            indice_atual_dem = indice_atual
            for i in range(indice_atual, np.shape(r)[0]):
                if (np.floor(r[i][0]) == np.floor(r[indice_atual_dem][0])) and (
                        np.floor(r[i][1]) == np.floor(r[indice_atual_dem][1])):
                    # testar achar a coordenada c0 do o x=0 e y=o e a partir dai achar pixel x1 e pixel yi pela diferença entre entre c0 e r[i]
                    # pixel_x1, pixel_y1 = int(r[i][1]-latdem0),int(r[i][0]-londem0)
                    pixel_x1, pixel_y1 = inv_transform * (r[i][0], r[i][1])
                    dist = distancia * i

                    #alt_dem = raster[int(pixel_y1)][int(pixel_x1)]
                    if int(pixel_x1)==3600 or int(pixel_y1) ==3600:
                        pixel_x1 = 3599
                        pixel_y1= 3599
                    alt_dem = raster[int(pixel_y1)][int(pixel_x1)]

                    # d.append(dist)
                    d[i] = dist
                    # dem.append(alt_dem)
                    dem[i] = alt_dem
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

                    #alt_dsm = raster_dsm[int(pixel_y1_dsm)][int(pixel_x1_dsm)]
                    if int(pixel_x1) == 3600 or int(pixel_y1) == 3600:
                        pixel_x1 = 3599
                        pixel_y1= 3599
                    alt_dsm = raster_dsm[int(pixel_y1_dsm)][int(pixel_x1_dsm)]

                    # dsm.append(alt_dsm)
                    dsm[i] = alt_dsm
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
                    # landcover.append(raster_landcover[int(pixel_y1_landcover)][int(pixel_x1_lancover)])
                    landcover[3 * i] = raster_landcover[int(pixel_y1_landcover)][int(pixel_x1_lancover)]
                    if i < np.shape(r)[0] - 1:
                        lonpasso = (r[i + 1][0] - r[i][0]) / 3
                        latpasso = (r[i + 1][1] - r[i][1]) / 3
                        pixel_x2_lancover, pixel_y2_landcover = inv_transform_landcover * (
                            r[i][0] + lonpasso, r[i][1] + latpasso)
                        pixel_x3_lancover, pixel_y3_landcover = inv_transform_landcover * (
                            r[i][0] + 2 * lonpasso, r[i][1] + 2 * latpasso)
                        if (np.floor(r[i][0] + 2 * lonpasso) == np.floor(r[indice_atual_land][0])) and (
                                np.floor(r[i][1] + 2 * latpasso) == np.floor(r[indice_atual_land][1])):
                            # landcover.append(raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)])
                            # landcover.append(raster_landcover[int(pixel_y3_landcover)][int(pixel_x3_lancover)])
                            landcover[3 * i + 1] = raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)]
                            landcover[3 * i + 2] = raster_landcover[int(pixel_y3_landcover)][int(pixel_x3_lancover)]
                        elif (np.floor(r[i][0] + lonpasso) == np.floor(r[indice_atual_land][0])) and (
                                np.floor(r[i][1] + latpasso) == np.floor(r[indice_atual_land][1])):
                            # landcover.append(raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)])
                            # landcover.append(raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)])
                            landcover[3 * i + 1] = raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)]
                            landcover[3 * i + 2] = raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)]
                        else:
                            # landcover.append(raster_landcover[int(pixel_y1_landcover)][int(pixel_x1_lancover)])
                            # landcover.append(raster_landcover[int(pixel_y1_landcover)][int(pixel_x1_lancover)])
                            landcover[3 * i + 1] = raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)]
                            landcover[3 * i + 2] = raster_landcover[int(pixel_y2_landcover)][int(pixel_x2_lancover)]
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


def perfil(p1, p2,  local_Configuracao, area=0):
    """Essa função extrai o perfil de elevação superfífice e land Cover ao longo do caminho entre dois pontos que
    estejam em raster diferentes. Ela usa a função obter_dados_do_raster para obter o perfil ao longo do caminho em
    um cada um dos Rasters e une os perfis de Rasters diferentes
    """
    indice_atual = 0
    # dem = []
    # dsm = []
    # landcover = []
    # d = []
    caminho, caminho_dsm, caminho_landcover = obter_raster(p1, p1)
    with rasterio.open(caminho) as src:
        transform = src.transform
        r, distancia = reta(p1, p2, transform[0])
    tamanho = len(r)
    dem = np.zeros(tamanho, dtype=float)
    dsm = np.zeros(tamanho, dtype=float)
    d = np.zeros(tamanho, dtype=float)
    landcover = np.zeros(3 * (tamanho - 1) + 1, dtype=int)
    while indice_atual < np.shape(r)[0] - 1:
        dem, dsm, landcover, d, indice_atual = obter_dados_do_raster(indice_atual, r, dem, dsm, landcover, d, distancia,
                                                                     area,  local_Configuracao)

    return dem, dsm, landcover, d, r


def raio_fresnel(n, d1, d2, f):
    """Função que calcula o raio da zona de Fresnel"""
    # f em hertz
    return (n * (c / f) * d1 * d2 / (d1 + d2)) ** 0.5


def obter_raster(ponto1, ponto2):  # (lon, lat)
    """Retorna o caminho do arquivo Raster que contém as coordenadas dos pontos dados"""
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
            if (int(lon1) % 3) == 0:
                raster_landcover = raster_landcover + we1 + str(int(lon1))
            else:
                raster_landcover = raster_landcover + we1 + str(int(lon1) + (3 - (int(lon1) % 3)))
        else:
            lon1_land = str(int(lon1) - (int(lon1) % 3))
            if len(lon1_land) == 3:
                raster_landcover = raster_landcover + we1 + lon1_land
            else:
                raster_landcover = raster_landcover + we1 + '0' + lon1_land
    elif len(lon1) == 2:
        raster1 = raster1 + we1 + '0' + lon1
        if we1 == 'W':
            if (int(lon1) % 3) == 0:
                raster_landcover = raster_landcover + we1 + '0' + str(int(lon1))
            else:
                raster_landcover = raster_landcover + we1 + '0' + str(int(lon1) + (3 - (int(lon1) % 3)))

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
                raster_landcover = raster_landcover + we1 + '00' + str(int(lon1))
            else:
                raster_landcover = raster_landcover + we1 + '00' + str(int(lon1) + (3 - (int(lon1) % 3)))
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

    return str(os.path.join('Raster', raster1 + '.tif')), str(
        os.path.join('dsm', raster1 + '.tif')), str(
        os.path.join('LandCover', raster_landcover + '.tif'))


def ajuste(elevacao, distancia, hg1, hg2, dl1, dl2):
    """Calculo ao ajuste linear de um perfil para obtençãio de Denta h e he1 e he2 parâmetreo do modelo ITM"""
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
    # z = np.array(zorig)
    # x = np.array(xorig)
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
    """A patrir de um perfil de terreno obtém os parametros do modelo ITM"""
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
    global Configuracao
    if Configuracao["urb"]:
        h_urb = abs((1 / Densidade_urbana) * (dsm[-1]+dsm[-2] - dem[-1]-dem[-2])/2)
    else:
        h_urb = 0

    return d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h_urb, visada, indice_visada_r, indice_visada


def obter_vegeta_atravessada(f, indice, dem, landcover, dsm, hr, ht, distancia, indice_d):
    """A partrir dos perfis obtém espessura da vegetação penetrada pelo sinal rádio"""
    dem = np.array(dem)
    dsm = np.array(dsm)

    altur_da_cobertuta = abs(dsm[indice:] - dem[indice:])
    espesura = 0
    if indice == 0:
        contar0 = 0
        rfresn3 = 0.6 * Modelos.raio_fresnel(1, distancia[-1] / 2, distancia[-1] / 2, f)
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
            if los[i] < ((rfresn3 * (abs(len(los) - 1) / 2 - i) * 2 / (len(los) - 1))):
                contar0 = contar0 + 1
            if los[i] < altur_da_cobertuta[i]:
                for n in (0, 1, 2):
                    if landcover[3 * (indice + i) + n] == 10:
                        espesura = espesura + 10  # ( colocar 5, metade dos 10 m)
        if (contar0 > 0) and (espesura > 100):
            espesura = espesura / 2


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
        x2 = np.array(distancia[:indice_d + 1])
        if c2 < 0:
            y2 = m2 * x2
            los2 = y2 - (dem[:indice_d + 1] - (dem[0]))
        else:
            y2 = m2 * x2 + c2
            los2 = y2 - (dem[:indice_d + 1] - (dem[indice_d]))
        contar1 = 0
        contar2 = 0
        for i in range(len(los) - 1):
            if los[i] < ((rfresn * (len(los) - i - 1) / (len(los) - 1))):
                contar1 = contar1 + 1
            if los[i] < altur_da_cobertuta[i]:
                for n in (0, 1, 2):
                    if landcover[3 * (indice + i) + n] == 10:
                        espesura = espesura + 10  # ( colocar 5, metade dos 10 m)

        if (contar1 > 0) and (espesura > 100):
            espesura = espesura / 2
        ref = espesura
        altur_da_cobertuta2 = abs(dsm[:indice_d + 1] - dem[:indice_d + 1])
        for i in range(len(los2) - 2):
            if los2[i] < ((rfresn2 * (len(los2) - i - 2) / (len(los2) - 2))):
                contar2 = contar2 + 1

            if los2[i] < altur_da_cobertuta2[i]:
                for n in (0, 1, 2):
                    if landcover[3 * i + n] == 10:
                        espesura = espesura + 10  # ( colocar 5, metade dos 10 m)
        if (contar2 > 0) and (espesura > 100):
            espesura = ref + (espesura - ref) / 2

    return 0.6 * espesura  # considerando 50% da area coberta com vegetação elevada. a documentação dos dados estabelec 10% ou mais


def addfoliun(local_mapas, local_cobertura):
    """Essa função utiliza a bibliotega Folium para criar o mapa e as camadas de visualização na tela principal do
    software """
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

    for i in local_mapas:
        carregamapa(i[0], i[1]).add_to(folium_map)

    for i in local_cobertura:
        criamapa(i['raster'], i['img'], local_cobertura).add_to(folium_map)

    folium_map.add_child(folium.LayerControl())
    return folium_map


# As funções abaixo usam a biblioteca Flask para gerar os Templates

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    global radios
    global Configuracao
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            if users[username] == password:
                session['username'] = username
                session['markers'] = []
                session['perdas'] = []
                session['cobertura'] = []
                session['Configuracao'] = Configuracao
                session['mapas'] = []
                session['radios'] = radios
                return redirect(url_for('home'))
            else:
                flash('Senha incorreta. Tente novamente.')
        else:
            return redirect(url_for('create_user', username=username))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/create_user', methods=['GET', 'POST'])
def create_user():
    global radios
    global Configuracao
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username not in users:
            users[username] = password
            session['username'] = username
            session['markers'] = []
            session['perdas'] = {}
            session['cobertura'] = []
            session['Configuracao'] = Configuracao
            session['mapas'] = []
            session['radios'] = radios
            return redirect(url_for('home'))
        else:
            flash('Usuário já existe. Tente novamente.')
    username = request.args.get('username', '')
    return render_template('create_user.html', username=username)


@app.route('/home')
def home():
    if 'username' in session:
        return redirect(url_for('index_map'))
    else:
        return redirect(url_for('login'))


@app.route('/index_map', methods=['GET', 'POST'])
def index_map():
    if 'username' not in session:
        return redirect(url_for('login'))

    fol = addfoliun(session['mapas'], session['cobertura'])
    fol.add_child(folium.LatLngPopup())

    for marker in session['markers']:
        folium.Marker([marker['lat'], marker['lon']]).add_to(fol)

    map_file = 'map.html'
    map_file_path = os.path.join("templates", map_file)
    fol.save(map_file_path)

    return render_template('index1.html', map_file=map_file)


@app.route('/addponto', methods=['GET', 'POST'])
def addponto():
    if 'username' not in session:
        return redirect(url_for('login'))
    rad = session['radios']
    return render_template('addponto.html', radios=rad)


@app.route('/add_marker', methods=['POST'])
def add_marker():
    if 'username' not in session:
        return redirect(url_for('login'))

    lat = float(request.form.get('lat'))
    lon = float(request.form.get('lon'))
    nome = str(request.form.get('nome'))
    ra = str(request.form.get('radio'))
    pot = float(request.form.get('pot'))
    h = float(request.form.get('h'))
    ant = str(request.form.get('ant'))

    local_markers = session['markers']
    local_markers.append({'lat': lat, 'lon': lon, 'nome': nome, 'h': h, 'radio': ra, 'pot': pot, 'ant': ant})
    session['markers'] = local_markers

    return jsonify({'result': 'success'})


@app.route('/ptp', methods=['GET', 'POST'])
def ptp():
    if 'username' not in session:
        return redirect(url_for('login'))

    local_markers = session['markers']
    local_Configuracao = session['Configuracao']
    local_radios = session['radios']
    local_perdas = {}
    fig_name = ''
    figura = ''
    ht, hr, potenciat, potenciar, sensibilidadet, sensibilidader, g1, g2 = 0, 0, 0, 0, 0, 0, 0, 0
    p1 = ()
    p2 = ()
    if request.method == "POST":

        # calcular perda Aqui antes das operacoes abaixo
        if request.form.get("ponto1") and request.form.get("ponto2") and request.form.get("f"):
            fig_name = str(request.form.get("ponto1")) + "_" + str(request.form.get("ponto2"))
            for i in local_markers:
                if i['nome'] == request.form.get("ponto1"):
                    p1 = (i['lon'], i['lat'])
                    ht = i['h']
                    potenciat = float(i['pot'])
                    ant = i['ant']
                    num = 0
                    for y in range(len(local_radios)):
                        if local_radios[y]['nome'] == i['radio']:
                            num = y
                    sensibilidadet = local_radios[num]['sensibilidade']
                    list_antenas = local_radios[num]['antenas']
                    for j in list_antenas:
                        if j['nome'] == ant:
                            g1 = j['ganho']

                elif i['nome'] == request.form.get("ponto2"):
                    p2 = (i['lon'], i['lat'])
                    hr = i['h']
                    potenciar = float(i['pot'])
                    ant = i['ant']
                    num = 0
                    for y in range(len(local_radios)):
                        if local_radios[y]['nome'] == i['radio']:
                            num = y
                    sensibilidader = local_radios[num]['sensibilidade']
                    list_antenas = local_radios[num]['antenas']
                    for j in list_antenas:
                        if j['nome'] == ant:
                            g2 = j['ganho']

            f = float(request.form.get("f"))
            dem, dsm, landcover, distancia, r_global = perfil(p1, p2,  local_Configuracao)
            Densidade_urbana = 1
            d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h_urb, visada, indice_visada_r, indice_visada = obter_dados_do_perfil(
                dem, dsm, distancia, ht, hr, Densidade_urbana)
            if (landcover[-1] == 50)or(landcover[-2] == 50) or (landcover[-3] == 50):
                urban = 'wi'
            else:
                urban = 'n'
            yt = 1  # é a perda pelo clima, adotar esse valor padrao inicialmente
            qs = 5  # 70% das situacões

            espesura = obter_vegeta_atravessada(f, indice_visada_r, dem, landcover, dsm, hr, ht, distancia,
                                                indice_visada)

            vegetacao = Modelos.atenuaca_vegetacao_antiga_ITU(f, espesura)

            print(ht)
            print(hr)
            print(espesura)
            print(vegetacao)

            hmed = (dem[0] + dem[-1]) / 2
            dls, hs = parametros_difracao(distancia, dem, hg1, hg2)
            espaco_livre = Modelos.friis_free_space_loss_db(f, d)
            epstein = Modelos.modelo_epstein_peterson(dls, hs, f)
            itm, variabilidade_situacao, At, dls_LR = Modelos.longLq_rice_model(hmed, f, hg1, hg2, he1, he2, d, yt, qs,
                                                                                dl1,
                                                                                dl2,
                                                                                Dh, visada,
                                                                                teta1, teta2, polarizacao='v')

            min_alt = Modelos.min_alt_ikegami(f)
            if h_urb > float(local_Configuracao["alt_max"]):
                h_urb = float(local_Configuracao["alt_max"]) + min_alt
            else:
                h_urb = h_urb + min_alt
            if (urban == 'wi'):
                if (h_urb > hg2 + min_alt):
                    urb = Modelos.ikegami_model(h_urb, hg2, f, w=float(local_Configuracao["largura_da_rua"]))
                else:
                    h_urb = hg2 + min_alt
                    urb = Modelos.ikegami_model(h_urb, hg2, f)
            else:
                urb = 0

            # colocar aqu uma funcao que adiciona a perda por vegetacao
            if (Dh > 90):
                Perda_por_terreno = (epstein)
            else:
                Perda_por_terreno = (itm + variabilidade_situacao)
            perda = Perda_por_terreno + vegetacao + urb + espaco_livre

            perda = round(10 * perda) / 10
            pott1 = round(10 * (10 * np.log10(1000 * potenciat))) / 10
            pott2 = round(10 * (10 * np.log10(1000 * potenciar))) / 10
            potr1 = round(10 * (pott2 + g1 + g2 - perda)) / 10
            potr2 = round(10 * (pott1 + g1 + g2 - perda)) / 10
            g1 = round(10 * g1) / 10
            g2 = round(10 * g2) / 10
            fateqp = min(pott1 + g1 + g2 - sensibilidader, pott2 + g1 + g2 - sensibilidadet)
            if fateqp - perda > 0:
                resultado = "Link fecha"
            else:
                resultado = "Link não fecha"
            if pott1 + g1 + g2 - sensibilidader==fateqp:
                potencia_dbw = pott1 + g1 + g2
                sensi_ref=sensibilidader
            else:
                potencia_dbw = pott2 + g1 + g2
                sensi_ref = sensibilidadet
            local_perdas = {'ponto1': request.form.get("ponto1"),
                            'ponto2': request.form.get("ponto2"),
                            'f': f,
                            'EspacoLivre': round(10 * espaco_livre) / 10,
                            'urb': round(10 * urb) / 10,
                            'veg ': round(10 * vegetacao) / 10,
                            'terreno': round(10 * Perda_por_terreno) / 10,
                            'perda': round(10 * perda) / 10,
                            'pott1': pott1,
                            'pott2': pott2,
                            'ganho1': g1,
                            'ganho2': g2,
                            'potr1': potr1,
                            'potr2': potr2,
                            'sensp1': sensibilidadet,
                            'sensp2': sensibilidader,
                            'resultado': resultado}

            vet_perdas = np.zeros(len(dem))
            for u in range(len(dem)):
                if u > 5:
                    d, hg1, hg2, dl1, dl2, teta1, teta2, he1, he2, Dh, h_urb, visada, indice_visada_r, indice_visada = obter_dados_do_perfil(
                        dem[:u + 1], dsm[:u + 1],
                        distancia[:u + 1], hg1, hg2,
                        Densidade_urbana)
                    if landcover[:3 * u + 1][-1] == 50 and landcover[:3 * u + 1][-2] == 50:
                        urban = 'wi'
                    else:
                        urban = 'n'
                    yt = 1  # é a perda pelo clima, adotar esse valor padrao inicialmente
                    qs = 5  # 70% das situacões
                    espesura = obter_vegeta_atravessada(f, indice_visada_r, dem[:u + 1], landcover[:3 * u + 1],
                                                        dsm[:u + 1], hg2, hg1, distancia[:u + 1], indice_visada)
                    # colocar a cidicao para chamar itm ou urbano + espaco livre

                    h0 = (dem[0] + dem[-1]) / 2

                    dls, hs = parametros_difracao(distancia[:u + 1], dem[:u + 1], hg1, hg2)

                    epstein = Modelos.modelo_epstein_peterson(dls, hs, f)
                    espaco_livre = Modelos.friis_free_space_loss_db(f, d)
                    itm, variabilidade_situacao, At, dLss = Modelos.longLq_rice_model(h0, f, hg1, hg2, he1, he2, d, yt,
                                                                                      qs, dl1, dl2, Dh, visada,
                                                                                      teta1, teta2, polarizacao='v',
                                                                                      simplificado=0)
                    if (((Dh > 90) and (d <= 0.7 * dls_LR))) or (d < 2000):
                        Perda_por_terreno = (epstein)
                    else:
                        Perda_por_terreno = (itm + variabilidade_situacao)

                    h_urb = h_urb + 0.5
                    if urban == 'wi' and h_urb > hg2 + 0.5:
                        urb = Modelos.ikegami_model(h_urb, hg2, f)
                    else:
                        urb = 0
                    vet_perdas[u] = itm
                    vegetacao = Modelos.atenuaca_vegetacao_antiga_ITU(f, espesura)
                    vet_perdas[u] = potencia_dbw - (vegetacao + urb + Perda_por_terreno + espaco_livre)

            fig, ax1 = plt.subplots()
            ax1.plot(distancia, dem, label='Perfil do terreno', color="blue")
            # ax1.plot(distancia, sperficie, label='Perfil do terreno', color="green")
            ax1.set_xlabel('Distância (m)')
            ax1.set_ylabel('Elevação do terreno (m)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax2 = ax1.twinx()

            ax2.plot(distancia, vet_perdas, label='Perfil do terreno', color="red")
            ax2.set_ylabel('Potência em dBm', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            titulo = 'Perfil do terreno ' + fig_name + ', e potência recebida'
            plt.title(titulo)
            fig.tight_layout()
            figura = "static/imagens/perfil_" + fig_name + ".jpg"
            # fig.figure(figsize=(10, 5))
            fig.savefig(figura, format="jpg")

    return render_template('ptp.html', perdas=local_perdas, markers=local_markers, figura=figura)


@app.route('/area', methods=['GET', 'POST'])
def area():
    if 'username' not in session:
        return redirect(url_for('login'))
    local_Configuracao = session['Configuracao']
    local_cobertura = session['cobertura']
    local_markers = session['markers']
    local_radios = session['radios']
    p1 = ()
    ht = 2
    potenciat, potenciar, sensibilidadet, sensibilidader, g1, g2 = 0, 0, 0, 0, 0, 0
    if request.form.get("ponto") and request.form.get("raio") and request.form.get("f") and request.form.get("ponto2"):
        for i in local_markers:
            if i['nome'] == request.form.get("ponto"):
                p1 = (i['lon'], i['lat'])
                ht = i['h']
                potenciat = float(i['pot'])
                ant = i['ant']
                num = 0
                for y in range(len(local_radios)):
                    if local_radios[y]['nome'] == i['radio']:
                        num = y
                sensibilidadet = local_radios[num]['sensibilidade']
                list_antenas = local_radios[num]['antenas']
                for j in list_antenas:
                    if j['nome'] == ant:
                        g1 = j['ganho']
            if i['nome'] == request.form.get("ponto2"):
                potenciar = float(i['pot'])
                ant = i['ant']
                num = 0
                for y in range(len(local_radios)):
                    if local_radios[y]['nome'] == i['radio']:
                        num = y
                sensibilidader = local_radios[num]['sensibilidade']
                list_antenas = local_radios[num]['antenas']
                for j in list_antenas:
                    if j['nome'] == ant:
                        g2 = j['ganho']

        pott1 = round(10 * (10 * np.log10(1000 * potenciat))) / 10
        pott2 = round(10 * (10 * np.log10(1000 * potenciar))) / 10

        limear = min(pott1 + g1 + g2 - sensibilidader, pott2 + g1 + g2 - sensibilidadet)
        hr = 1.5
        caminho, caminho_dsm, caminho_landcover = obter_raster(p1, p1)
        precisao = 1 / float(local_Configuracao[
                                 'precisao'])  # 0.5  # precisao 1=> grau em grau, precisao 2=> 0.5  em 0.5 graus, precição n=>1/n em 1/n graus
        largura_da_rua = local_Configuracao["largura_da_rua"]
        caminho = modificar_e_salvar_raster(caminho, p1, float(request.form.get("raio")), limear, ht, hr,
                                            float(request.form.get("f")), precisao, largura_da_rua, local_Configuracao)

        nova_cobertura = request.form.get("ponto") + '_Area_de_cobertura' + '_' + request.form.get("f")
        img = criaimg(caminho, nova_cobertura)
        novo = 0
        for i in local_cobertura:
            if i['nome'] == nova_cobertura:
                novo = 1

        if novo == 0:
            local_cobertura.append(
                {'nome': nova_cobertura, 'raster': caminho, 'f': float(request.form.get("f")), 'img': img,
                 'h': float(ht)})
        session['cobertura'] = local_cobertura

    return render_template('area.html')


@app.route('/config', methods=['GET', 'POST'])
def conf():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        local_Configuracao = {"urb": int(request.form.get("urb")), "veg": int(request.form.get("veg")),
                              "precisao": float(request.form.get("precisao")),
                              "largura_da_rua": float(request.form.get("larg")),
                              "alt_max": float(request.form.get("alt"))}
        session['Configuracao'] = local_Configuracao

    return render_template('conf.html')


@app.route('/addmapa', methods=['GET', 'POST'])
def addmapa():
    local_mapa = session['mapas']
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Nenhum arquivo enviado'

        file = request.files['file']

        if file.filename == '':
            return 'Nenhum arquivo selecionado'

        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            caminho_completo = os.path.join('uploads', filename)
            local_mapa.append([caminho_completo, filename[:-4]])
            session['mapas'] = local_mapa
            with Image.open(caminho_completo) as img:
                img = img.convert('RGB')
                img.save(caminho_completo[:-4] + '.jpg', 'JPEG')

            return 'Arquivo enviado com sucesso'
        else:
            return 'Arquivo não suportado'

    return render_template('addmapa.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return 'Nenhum arquivo enviado'

    file = request.files['file']

    if file.filename == '':
        return 'Nenhum arquivo selecionado'

    mapas = session['mapas']
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        caminho_completo = os.path.join('uploads', filename)
        mapas.append([caminho_completo, filename[:-4]])
        session['mapas'] = mapas
        with Image.open(caminho_completo) as img:
            img = img.convert('RGB')
            img.save(caminho_completo[:-4] + '.jpg', 'JPEG')

        return 'Arquivo enviado com sucesso'
    else:
        return 'Arquivo não suportado'


@app.route('/get_radio/<nome>')
def get_radio(nome):
    if 'username' not in session:
        return redirect(url_for('login'))

    rad = session['radios']
    for ra in rad:
        if ra['nome'] == nome:
            return jsonify({
                'potencia_tipo': ra['potencia']['tipo'],
                'potencia_valor': ra['potencia']['valor'],
                'antenas': [antena['nome'] for antena in ra['antenas']]
            })
    return jsonify({})


@app.route('/projetos', methods=['GET', 'POST'])
def projetos():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('projetos.html')


@app.route('/salv', methods=['GET', 'POST'])
def salv():
    if request.form.get("nsalv"):
        arquiv = "planejamentos\\" + str(request.form.get("nsalv"))
        arquiv = arquiv + ".pkl"
        markers = session['markers']
        perdas = session['perdas']
        cobertura = session['cobertura']
        local_Configuracao = session['Configuracao']
        mapas = session['mapas']
        radios = session['radios']
        data = {"markers": markers, "perdas": perdas, "cobertura": cobertura, "Configuracao": local_Configuracao,
                "mapas": mapas, "radios": radios}

        with open(arquiv, 'wb') as arquivo:
            # Salvar as variáveis no arquivo
            pickle.dump(data, arquivo)
    return redirect(url_for('projetos'))


@app.route('/carr', methods=['GET', 'POST'])
def carr():
    if request.form.get("ncarr"):
        arquiv = "planejamentos\\" + str(request.form.get("ncarr"))
        arquiv = arquiv + ".pkl"

        with open(arquiv, 'rb') as arquivo:
            # Carregar as variáveis do arquivo
            data = pickle.load(arquivo)
        session['markers'] = data['markers']
        session['perdas'] = data['perdas']
        session['cobertura'] = data['cobertura']
        session['Configuracao'] = data['Configuracao']
        session['mapas'] = data['mapas']
        session['radios'] = data['radios']

    return redirect(url_for('index_map'))


if __name__ == '__main__':
    app.run(debug=True)

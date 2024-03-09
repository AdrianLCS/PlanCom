from osgeo import gdal
import numpy as np

def obter_valor_dem(arquivo_dem, ponto_x, ponto_y):
    dataset = gdal.Open(arquivo_dem)
    banda_dem = dataset.GetRasterBand(1)
    transformacao = dataset.GetGeoTransform()

    # Convertendo coordenadas para índices de pixel
    coluna = int((ponto_x - transformacao[0]) / transformacao[1])
    linha = int((ponto_y - transformacao[3]) / transformacao[5])

    # Lendo o valor do DEM no ponto especificado
    valor_dem = banda_dem.ReadAsArray(coluna, linha, 1, 1)[0, 0]

    return valor_dem

def calcular_distancia_dem(arquivo_dem, ponto1, ponto2):
    valor_dem_ponto1 = obter_valor_dem(arquivo_dem, ponto1[0], ponto1[1])
    valor_dem_ponto2 = obter_valor_dem(arquivo_dem, ponto2[0], ponto2[1])

    diferenca_altitude = valor_dem_ponto2 - valor_dem_ponto1

    # Calcular distância horizontal considerando a resolução do DEM
    distancia_horizontal = ((ponto2[0] - ponto1[0])**2 + (ponto2[1] - ponto1[1])**2)**0.5

    # Calcular distância 3D usando o teorema de Pitágoras
    distancia_3d = (distancia_horizontal**2 + diferenca_altitude**2)**0.5

    return distancia_3d

# Exemplo de uso
arquivo_dem = "C:\PythonFlask\PlanCom\Raster\S30W052.tif"
ponto1 = (-43.220238,  -22.963526)  # coordenadas do ponto 1
ponto2 = (-43.1661, -22.9555)  # coordenadas do ponto 2

distancia = calcular_distancia_dem(arquivo_dem, ponto1, ponto2)
print(f'Distância entre os pontos: {distancia} metros')

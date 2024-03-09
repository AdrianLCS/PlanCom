import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import numpy as np

import rasterio
import numpy as np

def obter_valor_dem(arquivo_dem, ponto_x, ponto_y):
    with rasterio.open(arquivo_dem) as src:
        # Convertendo coordenadas para índices de pixel
        col, row = src.index(ponto_x, ponto_y)

        # Lendo o valor do DEM no ponto especificado
        valor_dem = src.read(1, window=((row, row+1), (col, col+1)))

    return valor_dem[0, 0]

def calcular_distancia_dem(arquivo_dem, ponto1, ponto2):
    with rasterio.open(arquivo_dem) as src:
        # Transformação afim do raster
        transformacao = src.transform

        valor_dem_ponto1 = obter_valor_dem(arquivo_dem, ponto1[0], ponto1[1])
        valor_dem_ponto2 = obter_valor_dem(arquivo_dem, ponto2[0], ponto2[1])

        diferenca_altitude = valor_dem_ponto2 - valor_dem_ponto1
        col, row = src.index(ponto1[0], ponto1[1])
        col1, row1 = src.index(ponto2[0], ponto2[1])
        # Calcular distância horizontal usando a escala do raster
        distancia_horizontal = np.sqrt((col1 - col)**2 + (row1 - row)**2) * transformacao[0]*(10**5)

        # Calcular distância 3D usando o teorema de Pitágoras
        distancia_3d = np.sqrt(distancia_horizontal**2 + diferenca_altitude**2)

        return distancia_3d

# Exemplo de uso
arquivo_dem = "C:\PythonFlask\PlanCom\Raster\S23W044.tif"
ponto1 = (-43.220238,  -22.963526)  # coordenadas do ponto 1
ponto2 = (-43.1661, -22.9555)   # coordenadas do ponto 2

distancia = calcular_distancia_dem(arquivo_dem, ponto1, ponto2)
print(f'Distância entre os pontos: {distancia} metros')
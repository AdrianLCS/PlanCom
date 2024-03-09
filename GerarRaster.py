import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# Coordenadas do centro do círculo
center_latitude = -26.5
center_longitude = -48.5

# Raio do círculo em quilômetros
radius_km = 10.0

# Nome do arquivo DEM de origem
input_dem_file = "C:\PythonFlask\PlanCom\Raster\S27\ASTGTMV003_S27W049_dem.tif"

# Nome do arquivo de saída
output_dem_file = "C:\PythonFlask\PlanCom\Raster\S27\Remake_S27W049_dem.tif"

# Abrir o DEM de origem para obter informações geográficas
with rasterio.open(input_dem_file) as src:
    # Obter informações sobre o DEM
    profile = src.profile
    transform = src.transform

    ###OBTER O PIXEL TENDO AS COORDENADAS
    inv_transform = ~src.transform
    # Especifica a latitude e a longitude desejadas
    latitude, longitude = (-26.5, -48.5)
    # Calcula as coordenadas do pixel correspondentes
    pixel_x, pixel_y = inv_transform * (longitude, latitude)


    # Criar uma matriz de zeros para o novo DEM
    new_dem = np.zeros_like(src.read(1))

    # Calcular as coordenadas do centro do círculo na matriz do DEM
    center_col, center_row = src.index(center_longitude, center_latitude)

    # Calcular o raio em pixels
    radius_pixels = int(radius_km / abs(transform.a))
    
    # Criar uma máscara circular
    y, x = np.ogrid[:src.height, :src.width]
    mask = (x - center_col) ** 2 + (y - center_row) ** 2 <= radius_pixels ** 2

    # Preencher o círculo com valores de elevação azuis (por exemplo, 0)
    new_dem[mask] = 100  # Ajuste conforme necessário

# Salvar o novo DEM como GeoTIFF
profile.update(count=1)  # Atualizar o número de bandas
with rasterio.open(output_dem_file, 'w', **profile) as dst:
    dst.write(new_dem, 1)

# Visualizar o novo DEM
plt.imshow(new_dem, cmap='terrain', extent=src.bounds)
plt.title('DEM com Círculo Azul')
plt.show()

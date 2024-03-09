import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

# Caminho para o arquivo DEM
dem_path = "C:\PythonFlask\PlanCom\Raster\S31\Teste3x3 - Copia.tif"

# Abre o raster DEM para leitura e escrita
with rasterio.open(dem_path, 'r+') as dem:

    # Obtem as informações do raster
    dem_profile = dem.profile

    # Defina as coordenadas do pixel que você deseja editar
    pixel_row = 100
    pixel_col = 150

    # Obtenha o valor original do pixel
    original_value = dem.read(1, window=((pixel_row, pixel_row + 1), (pixel_col, pixel_col + 1)))

    # Novo valor que você deseja atribuir ao pixel
    novo_valor = 1000

    # Atualize o valor do pixel
    dem.write(novo_valor, 1, window=((pixel_row, pixel_row + 1), (pixel_col, pixel_col + 1)))

    # Salve as mudanças
    dem.close()
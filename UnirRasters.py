import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import numpy as np

# Nomes dos arquivos dos quatro rasters
raster1 = 'C:\PythonFlask\PlanCom\Raster\S31\ASTGTMV003_S31W052_dem.tif'
raster2 = 'C:\PythonFlask\PlanCom\Raster\S31\ASTGTMV003_S31W051_dem.tif'
raster3 = 'C:\PythonFlask\PlanCom\Raster\S31\ASTGTMV003_S32W052_dem.tif'
raster4 = 'C:\PythonFlask\PlanCom\Raster\S31\ASTGTMV003_S32W051_dem.tif'

# Nome do arquivo de saída
output_raster = "C:\PythonFlask\PlanCom\Raster\S31\Teste.tif"

# Abrir os quatro rasters
with rasterio.open(raster1) as src1, rasterio.open(raster2) as src2, \
        rasterio.open(raster3) as src3, rasterio.open(raster4) as src4:

    # Obter informações sobre um dos rasters (usaremos raster1 como referência)
    profile = src1.profile
    transform = src1.transform

    # Obter os dados raster de cada arquivo
    data1 = src1.read(1)
    data2 = src2.read(1)
    data3 = src3.read(1)
    data4 = src4.read(1)
    #print(data4.shape)
    # Criar um array 2D para armazenar os dados combinados
    #combined_data = np.zeros_like(data1)
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

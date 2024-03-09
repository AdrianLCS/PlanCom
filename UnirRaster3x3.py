import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import numpy as np

# Nomes dos arquivos dos nove rasters
raster_files = [
    "C:\PythonFlask\PlanCom\Raster\S30\ASTGTMV003_S30W053_dem.tif", "C:\PythonFlask\PlanCom\Raster\S30\ASTGTMV003_S30W052_dem.tif", "C:\PythonFlask\PlanCom\Raster\S30\ASTGTMV003_S30W051_dem.tif",
    "C:\PythonFlask\PlanCom\Raster\S31\ASTGTMV003_S31W053_dem.tif", "C:\PythonFlask\PlanCom\Raster\S31\ASTGTMV003_S31W052_dem.tif", "C:\PythonFlask\PlanCom\Raster\S31\ASTGTMV003_S31W051_dem.tif",
    "C:\PythonFlask\PlanCom\Raster\S31\ASTGTMV003_S32W053_dem.tif", "C:\PythonFlask\PlanCom\Raster\S31\ASTGTMV003_S32W052_dem.tif", "C:\PythonFlask\PlanCom\Raster\S31\ASTGTMV003_S32W051_dem.tif"
]

# Nome do arquivo de saída
output_raster = "C:\PythonFlask\PlanCom\Raster\S31\Teste3x3.tif"

# Número de rasters ao longo de uma linha ou coluna
num_rasters_row_col = 3

# Abrir os nove rasters

with rasterio.open(raster_files[0]) as src0,rasterio.open(raster_files[1]) as src1,rasterio.open(raster_files[2]) as src2,\
        rasterio.open(raster_files[3]) as src3,rasterio.open(raster_files[4]) as src4,rasterio.open(raster_files[5]) as src5,\
        rasterio.open(raster_files[6]) as src6,rasterio.open(raster_files[7]) as src7,rasterio.open(raster_files[8]) as src8:
    srcs=[]
    srcs.append(src0), srcs.append(src1), srcs.append(src2), srcs.append(src3), srcs.append(src4)
    srcs.append(src5), srcs.append(src6), srcs.append(src7), srcs.append(src8)
    # Obter informações sobre um dos rasters (usaremos o primeiro como referência)
    profile = srcs[0].profile
    transform = srcs[0].transform

    # Criar um array 2D para armazenar os dados combinados
    combined_data = np.zeros((srcs[0].height * num_rasters_row_col, srcs[0].width * num_rasters_row_col), dtype=np.float32)

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

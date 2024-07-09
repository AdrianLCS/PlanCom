import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import numpy as np

# Nomes dos arquivos dos nove rasters

def generate_raster_files(base_string):
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


# Exemplo de uso
base_string = "Raster\\S31W052.tif"
raster_files = generate_raster_files(base_string)
print(raster_files)




def unir_raster_3x3(raster_path):

    raster_files = raster_files = generate_raster_files(raster_path)


    # Nome do arquivo de saída
    output_raster = "C:\PythonFlask\PlanCom\\"+ raster_path[:-4] + "_3x3.tif"

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
unir_raster_3x3(base_string)
from osgeo import gdal

# Caminho para o arquivo .hdf
file_path = 'C:\PythonFlask\PlanCom\LandCover\MCD12Q1.A2022001.h13v11.061.2023243160111.hdf'

# Abre o arquivo HDF
dataset = gdal.Open(file_path, gdal.GA_ReadOnly)

if dataset is not None:
    # Obtém o número de bandas no conjunto de dados
    num_bands = dataset.RasterCount

    # Lê os dados da primeira banda
    band = dataset.GetRasterBand(1)
    land_cover_data = band.ReadAsArray()

    # Fecha o conjunto de dados
    dataset = None

    # Agora você pode trabalhar com os dados conforme necessário
    print(land_cover_data)
else:
    print("Erro ao abrir o arquivo HDF.")
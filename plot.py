import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import geopandas as gpd

def obter_elevacao(lon, lat, raster_dataset):
    """
    Obtém a elevação de um ponto no conjunto de dados raster.

    Parameters:
    lon (float): Longitude do ponto.
    lat (float): Latitude do ponto.
    raster_dataset (rasterio.DatasetReader): Conjunto de dados raster.

    Returns:
    elevacao (float): Elevação do ponto.
    """
    lon, lat = raster_dataset.xy(lat, lon)
    for val in raster_dataset.sample([(lon, lat)]):
        elevacao = val.item()
    return elevacao

def obter_perfil_elevacao(ponto_inicial, ponto_final, raster_dataset, num_pontos=1000):
    """
    Obtém o perfil de elevação entre dois pontos usando o conjunto de dados raster.

    Parameters:
    ponto_inicial (tuple): Coordenadas (lon, lat) do ponto inicial.
    ponto_final (tuple): Coordenadas (lon, lat) do ponto final.
    raster_dataset (rasterio.DatasetReader): Conjunto de dados raster.
    num_pontos (int): Número de pontos ao longo do perfil.

    Returns:
    perfil_elevacao (list): Lista de elevações ao longo do perfil.
    """
    line = LineString([ponto_inicial, ponto_final])
    pontos_perfil = zip(*line.xy)
    elevacoes = [obter_elevacao(lon, lat, raster_dataset) for lon, lat in pontos_perfil]
    perfil_elevacao = elevacoes[:num_pontos]
    return perfil_elevacao

def plotar_perfil_elevacao(perfil_elevacao):
    """
    Plota o perfil de elevação.

    Parameters:
    perfil_elevacao (list): Lista de elevações ao longo do perfil.
    """
    plt.plot(perfil_elevacao)
    plt.title('Perfil de Elevação')
    plt.xlabel('Distância ao Longo do Perfil')
    plt.ylabel('Elevação (metros)')
    plt.show()

def main():
    # Substitua 'seu_arquivo_raster.tif' pelo caminho do seu arquivo raster GMTED2010S30W060.
    arquivo_raster = "C:\PythonFlask\PlanCom\Raster\sudeste\Teste.tif"

    # Substitua as coordenadas pelos pontos inicial e final desejados.
    ponto_inicial = (-43.6333, -19.4505)  # São Paulo
    ponto_final = (-46.6031, -22.5615)  # São Paulo

    # Abre o conjunto de dados raster com rasterio.
    with rasterio.open(arquivo_raster) as raster_dataset:
        # Obtém o perfil de elevação entre os dois pontos.
        perfil_elevacao = obter_perfil_elevacao(ponto_inicial, ponto_final, raster_dataset)

        # Plota o perfil de elevação.
        plotar_perfil_elevacao(perfil_elevacao)

if __name__ == "__main__":
    main()
import numpy as np
import rasterio
import folium
import matplotlib.pyplot as plt
import ee

ee.Initialize(project="plancom-409417")

from rasterio.transform import from_origin




def criaimg(dem_file):
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
    plt.savefig(dem_file[:-3] + "png", format="png", bbox_inches='tight', pad_inches=0)

    # Fechar a figura para liberar recursos
    plt.close()
    return dem_file[:-3] + "png"


def criamapa(raster_path, ponto, raio, limear):
    # Carregar o arquivo DEM (tif)
    dem_file2 = criaimg(raster_path)
    dem_dataset = rasterio.open(raster_path)

    # Obter as informações sobre a extensão do DEM
    bounds = dem_dataset.bounds
    min_lat, min_lon = bounds.bottom, bounds.left
    max_lat, max_lon = bounds.top, bounds.right

    # Calcular o centro do DEM para definir o local inicial do mapa
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Criar um mapa OpenStreetMap usando Folium
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')

    # Adicionar a camada do raster como uma sobreposição de imagem
    image_overlay = folium.raster_layers.ImageOverlay(
        image=dem_file2,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.5,
        name='DEM Overlay',
        interactive=True,
        cross_origin=False,
        zindex=1
    )

    image_overlay.add_to(m)

    # Adicionar uma camada de controle de camadas ao mapa
    folium.LayerControl().add_to(m)

    # Salvar o mapa como um arquivo HTML
    m.save("mapa_com_dem_overlay.html")


# Caminho para o arquivo raster DEM original
raster_path = 'C:\PythonFlask\PlanCom\Raster\S23W044.tif'

# Coordenadas do ponto a ser modificado (linha, coluna)
ponto = (-43.1895, -22.9036)

# Novo valor para o ponto
raio = 5000
limear = 0

criamapa(raster_path, ponto, raio, limear)

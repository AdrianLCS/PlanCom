import elevatr
import geopandas as gpd
import matplotlib.pyplot as plt

def get_elevation_profile(start_point, end_point, samples=100):
    # Criação de um GeoDataFrame com os pontos iniciais e finais
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([start_point[0], end_point[0]], [start_point[1], end_point[1]]), crs="EPSG:4326")

    # Obtendo dados de elevação para o perfil do terreno
    elevation_data = elevatr.point_query(gdf.geometry, method="srtm", samples=samples)

    # Extraindo as coordenadas e altitudes do resultado
    coords = gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
    altitudes = [entry[0]['elevation'] for entry in elevation_data]

    return coords, altitudes

def plot_elevation_profile(coords, altitudes):
    plt.plot(altitudes)
    plt.title('Perfil de Elevação do Terreno')
    plt.xlabel('Amostras')
    plt.ylabel('Altitude (metros)')
    plt.xticks(range(len(coords)), coords, rotation=45)
    plt.show()

# Exemplo de uso:
start_point = (-22.9068, -43.1729)  # Coordenadas do Rio de Janeiro
end_point = (-23.5505, -46.6333)    # Coordenadas de São Paulo

coords, altitudes = get_elevation_profile(start_point, end_point)
plot_elevation_profile(coords, altitudes)
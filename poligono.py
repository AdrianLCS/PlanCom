import folium

# Coordenadas do polígono (pode ser uma lista de coordenadas)
coordenadas_poligono = [
    (-23.5505, -46.6333),
    (-23.5550, -46.6250),
    (-23.5600, -46.6400)
]

# Criando um objeto de mapa Folium
mapa = folium.Map(location=[-23.5505, -46.6333], zoom_start=14)

# Adicionando o polígono ao mapa
folium.Polygon(locations=coordenadas_poligono, color='blue', fill=True, fill_color='lightblue', fill_opacity=0.5).add_to(mapa)

# Exibindo o mapa
mapa.save("mapa_com_poligono_desenhado.html")
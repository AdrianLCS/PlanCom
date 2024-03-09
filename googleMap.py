from flask import Flask, render_template
import ee
import folium

app = Flask(__name__)

# Configuração da API do Google Earth Engine
ee.Initialize(project="plancom-409417")

# Função para obter a camada de elevação
def get_static_elevation_image():
    # Substitua isso pelo código específico da sua região ou área de interesse
    area_of_interest = ee.Geometry.Rectangle([-180, -90, 180, 90])

    # Crie uma imagem de elevação usando a API do Google Earth Engine
    elevation = ee.Image("USGS/SRTMGL1_003").clip(area_of_interest)

    # Converta a imagem de elevação em uma imagem estática (por exemplo, PNG)
    elevation_static = elevation.getThumbURL({'dimensions': 512, 'format': 'png'})

    return elevation_static

# Rota principal que renderiza o mapa com a camada de elevação
@app.route('/')
def index():
    elevation_static = get_static_elevation_image()

    # Crie um mapa Leaflet
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=8)

    # Adicione a camada de elevação estática ao mapa Leaflet como uma imagem
    folium.raster_layers.ImageOverlay(
        image=elevation_static,
        bounds=[[37.7749 - 1, -122.4194 - 1], [37.7749 + 1, -122.4194 + 1]],
        opacity=1.0
    ).add_to(m)

    # Salve o mapa como um arquivo HTML temporário
    m.save("templates/map.html")

    # Renderize o arquivo HTML
    return render_template('map.html')

if __name__ == '__main__':
    app.run(debug=True)
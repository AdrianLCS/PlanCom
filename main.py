from flask import Flask, render_template, request
import folium
import rasterio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/plot', methods=['POST'])
def plot():
    # Obter as coordenadas dos pontos marcados pelo cliente
    lat1 = float(request.form['lat1'])
    lon1 = float(request.form['lon1'])
    lat2 = float(request.form['lat2'])
    lon2 = float(request.form['lon2'])

    # Carregar o conjunto de dados SRTM
    raster_file = "C:\PythonFlask\PlanCom\Raster\sudeste\Teste.tif"  # Substitua pelo caminho real para seu arquivo raster
    with rasterio.open(raster_file) as src:
        # Obter elevações ao longo da linha entre os pontos
        elevations = src.sample([(lon1, lat1), (lon2, lat2)])

    # Criar um gráfico 3D do perfil de elevação
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([lon1, lon2], [lat1, lat2], elevations.flatten(), marker='o')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevação (metros)')

    # Converter o gráfico em uma imagem para exibir no navegador
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)

    # Converter a imagem em base64 para incorporar em uma tag <img> no HTML
    img_base64 = base64.b64encode(img_data.read()).decode('utf-8')

    return render_template('plot.html', img_base64=img_base64)


if __name__ == '__main__':
    app.run(debug=True)

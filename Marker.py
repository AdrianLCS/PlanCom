from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Lista para armazenar os marcadores
markers = []

@app.route('/')
def index():
    return render_template('index.html', markers=markers)

@app.route('/add_marker', methods=['POST'])
def add_marker():
    # Obtemos as coordenadas do marcador do corpo da solicitação
    lat = float(request.form.get('lat'))
    lon = float(request.form.get('lon'))

    # Adicionamos o marcador à lista
    markers.append({'lat': lat, 'lon': lon})

    return jsonify({'result': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
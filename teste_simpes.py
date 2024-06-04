from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    image_path = 'static/images/perfil_PDC_IME.jpg'  # Caminho relativo para a imagem na pasta 'static'
    return render_template('index.html', image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)

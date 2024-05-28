class MeuObjeto:
    def __init__(self, nome, opcoes):
        self.nome = nome
        self.opcoes = opcoes

# Lista de objetos
lista = [
    MeuObjeto("Objeto1", ["Opção1", "Opção2", "Opção3"]),
    MeuObjeto("Objeto2", []),
    MeuObjeto("Objeto3", ["OpçãoA", "OpçãoB"])
]


from flask import Flask, render_template, jsonify



app = Flask(__name__)

# Definindo a classe e a lista de objetos
class MeuObjeto:
    def __init__(self, nome, opcoes):
        self.nome = nome
        self.opcoes = opcoes

lista = [
    MeuObjeto("Objeto1", ["Opção1", "Opção2", "Opção3"]),
    MeuObjeto("Objeto2", []),
    MeuObjeto("Objeto3", ["OpçãoA", "OpçãoB"])
]

@app.route('/')
def index():
    return render_template('index.html', objetos=lista)

@app.route('/get_opcoes/<nome>')
def get_opcoes(nome):
    for obj in lista:
        if obj.nome == nome:
            return jsonify(obj.opcoes)
    return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)
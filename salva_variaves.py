import pickle

# Suponha que você tenha algumas variáveis que deseja salvar
variavel1 = [1, 2, 3, 4, 5]
variavel2 = {"chave1": "valor1", "chave2": "valor2"}

# Abrir um arquivo para escrita binária
with open('variaveis.pkl', 'wb') as arquivo:
    # Salvar as variáveis no arquivo
    pickle.dump(variavel1, arquivo)
    pickle.dump(variavel2, arquivo)

################CARREGAR##############

# Abrir o arquivo para leitura binária
with open('variaveis.pkl', 'rb') as arquivo:
    # Carregar as variáveis do arquivo
    variavel1_carregada = pickle.load(arquivo)
    variavel2_carregada = pickle.load(arquivo)

print(variavel1_carregada)
print(variavel2_carregada)

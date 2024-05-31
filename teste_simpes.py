import matplotlib.pyplot as plt

# Dados para a primeira curva
x1 = [0, 1, 2, 3, 4, 5]
y1 = [0, 1, 4, 9, 16, 25]

# Dados para a segunda curva
x2 = [0, 1, 2, 3, 4, 5]
y2 = [0, 11, 30, 40, 60, 15]

# Dados para a terceira curva
x3 = [0, 1, 2, 3, 4, 5]
y3 = [0, 100, 200, 300, 400, 500]

# Criar o gráfico
fig, ax1 = plt.subplots()

# Primeira curva no eixo y primário
ax1.plot(x1, y1, label='Curva 1', color='blue')
ax1.set_xlabel('Eixo X')
ax1.set_ylabel('Curva 1', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Criar um segundo eixo y compartilhando o mesmo eixo x
ax2 = ax1.twinx()

# Segunda curva no eixo y secundário
ax2.plot(x2, y2, label='Curva 2', color='red')
ax2.set_ylabel('Curva 2', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Criar um terceiro eixo y compartilhando o mesmo eixo x
ax3 = ax1.twinx()

# Posição do terceiro eixo y
#ax3.spines['right'].set_position(('outward', 60))
ax3.plot(x3, y3, label='Curva 3', color='green')
#ax3.set_ylabel('Curva 3', color='green')
ax3.tick_params(axis='y', labelcolor='green')

# Adicionar título e ajustar o layout
plt.title('Exemplo de Gráfico com Três Curvas e Escalas Diferentes')
fig.tight_layout()  # Para ajustar o layout e evitar sobreposição

# Mostrar o gráfico
plt.show()

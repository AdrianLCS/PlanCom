"""Para executar um código MATLAB a partir do Python e acessar suas variáveis, você pode usar a biblioteca matlab.engine. Primeiro, certifique-se de ter o MATLAB Engine API para Python instalado no seu sistema.

Aqui está um exemplo de como você pode fazer isso:

Instalar o MATLAB Engine API para Python:

Você pode seguir as instruções fornecidas pela MathWorks para instalar o MATLAB Engine API para Python: Install MATLAB Engine API for Python

Exemplo de código Python:

Depois de instalar o MATLAB Engine API, você pode criar uma ponte entre o Python e o MATLAB. Aqui está um exemplo:"""

import matlab.engine

# Iniciar o MATLAB Engine
eng = matlab.engine.start_matlab()

# Executar código MATLAB
matlab_code = """
A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
B = A * 2;
"""

eng.eval(matlab_code, nargout=0)

# Acessar variáveis MATLAB no Python
A = eng.workspace['A']
B = eng.workspace['B']

# Exibir as variáveis no Python
print("Matriz A:")
print(A)
print("\nMatriz B:")
print(B)

# Encerrar o MATLAB Engine
eng.quit()
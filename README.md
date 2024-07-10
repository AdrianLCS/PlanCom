# ComPlanner

# Tema:

SOFTWARE DE PLANEJAMENTO DE MEIOS DE COMUNICAÇÕES COM BASE EM CARTAS DIGITALIZADAS DE UM AMBIENTE OPERACIONAL

# Conceito:

Sofrware que calcula a preda devido ao terreno, à vegetação e à fatores urbanos com base no modelo ITM, Ikegami e recomendações ITU. O usuário interage com um mapa, onde ele adiciona marcadores e equpamentos rádio e o sofrware retorna a preda de enlace e a informação se será ou não possível a comunicaçõa entre os dois pontos marcados.

# Função:

Cálculo da atenuação rádio.

# Motivação:

[...]
# Tutorial de execução do script:
  1- Instale o interpretador python e adicione python nas variáveis de ambiente. 

  2- No ambiente com python instalado, instale a biblioteca Rasterio:  "python -m pip install rasterio"
    
  3- No ambiente com python instalado, instale a biblioteca Pillow com o comando: "python -m pip install pillow"
  
  4- No ambiente com python instalado, instale a biblioteca Flask:  "python -m pip install flask"

  5- No ambiente com python instalado, instale a biblioteca Folium:  "python -m pip install folium"

  6- Baixe o toda a pasta do projeto programa e rode o arquivo main.py: Abra o navegador e acessa a máquina em que o software está rodando pelo seu endreço IP na porta 5000.

# Tutorial de Uso:


A pagina inicial "Aba home"do app é demonstrada abaixo, nela o usuário pode visulizar os mapas, potos adicionos e camada de cobertura rádio:


![index](https://github.com/AdrianLCS/PlanCom/assets/114261968/6a4c2c03-113b-4422-bbed-cc1a9380c17c)

A Figura abaixo mostra a aba "Add Ponto", onde é realizada a adição de marcadores no mapa. Nessa aba, o usuário pode adicionar um ponto que vai aparecer no mapa através de um marcador. O usuário entra com o nome que deseja dar ao ponto, as coordenadas e a altura da antena, e seleciona o equipamento rádio que será operado nesse ponto. Ao preencher os campos e clicar no botão "Adicionar Marcador", aparecerá um marcador no mapa nas referidas coordenadas. A Figura da aba home acima mostra dois marcadores referentes aos locais do IME e do PCD que foram adicionados

![addpont](https://github.com/ProgramacaoAplicada2022/Adrian_Willian_Stegnographer/assets/114261968/c61e3f92-87f9-420d-b48d-02403a102ae4)


[...]
[...]
[...]
# Tutorial de compliação do código:
[...]

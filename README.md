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
# Arqivos necessários:
Para cada Região de planejamento é necessário baixar o Modelo digital de superfíce DSM precição 1 arcsec o modelo digita de Terreno DTM  precisão 1 arcsec e o modelo de digital de Land Cover precição 10 m.
Verifique quais a coordenadas do ponto onde se deseja fazer a predião de enlace rádio e baixe os modelos de Elevação e land Cover que abranjam esse local.

# Local de Download dos modelos digitais de Elevação e Land Cover:
O DSM pode ser baixado no link: https://opentopography.s3.sdsc.edu/minio/raster/AW3D30/AW3D30_global/ ou https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3.
O DTM pode ser baixado no link: https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn
O LandCover pode ser baixado no link: https://worldcover2021.esa.int/downloader
Todos em formato .tif

# Local onde os modelos digitais de Elevação e Land Cover devem ser colocados:
DTM:
Copie e cole o DTM dentro da pasta Raster que está na pasta do projeto com o nome padrão S04W054.tif sendo 04 a latitude em graus do ponto mais a sul do arquivo e 054 a longitude em graus do ponto mais a oeste do arquivo.
DSM:
Copie e cole o DSM dentro da pasta dsm que está na pasta do projeto com o nome padrão, por exempo N05E047.tif sendo 05 a latitude em graus do ponto mais a sul do arquivo e 057 a longitude em graus do ponto mais a oeste do arquivo.
LandCover:
Copie e cole o arqivo de Land Cover dentro da pasta LandCover que está na pasta do projeto com o nome padrão, por exempo N12W021.tif sendo 12 a latitude em graus do ponto mais a sul do arquivo e 021 a longitude em graus do ponto mais a oeste do arquivo.


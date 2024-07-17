# ComPlanner

# Tema:

SOFTWARE DE PLANEJAMENTO DE MEIOS DE COMUNICAÇÕES COM BASE EM CARTAS DIGITALIZADAS DE UM AMBIENTE OPERACIONAL

# Conceito:

Software que calcula a perda devido ao terreno, à vegetação e a fatores urbanos com base no modelo ITM, Ikegami e recomendações ITU. O usuário interage com um mapa, onde ele adiciona marcadores e equipamentos rádio e o software retorna a perda de enlace e a informação se será ou não possível a comunicação entre os dois pontos marcados.

# Função:

Cálculo da atenuação rádio.

# Motivação:

[...]

# Tutorial de execução do script:
1. Instale o interpretador Python e adicione Python nas variáveis de ambiente.

2. No ambiente com Python instalado, instale a biblioteca Rasterio: `python -m pip install rasterio`

3. No ambiente com Python instalado, instale a biblioteca Pillow com o comando: `python -m pip install pillow`

4. No ambiente com Python instalado, instale a biblioteca Flask: `python -m pip install flask`

5. No ambiente com Python instalado, instale a biblioteca Folium: `python -m pip install folium`

6. Baixe toda a pasta do projeto e rode o arquivo `main.py`. Abra o navegador e acesse a máquina em que o software está rodando pelo seu endereço IP na porta 5000.

# Tutorial de Uso:

A página inicial "Aba home" do app é demonstrada abaixo, nela o usuário pode visualizar os mapas, pontos adicionados e camada de cobertura rádio:

![index](https://github.com/AdrianLCS/PlanCom/assets/114261968/6a4c2c03-113b-4422-bbed-cc1a9380c17c)

Na aba "Add Ponto", onde é realizada a adição de marcadores no mapa. Nessa aba, o usuário pode adicionar um ponto que vai aparecer no mapa através de um marcador. O usuário entra com o nome que deseja dar ao ponto, as coordenadas e a altura da antena, e seleciona o equipamento rádio que será operado nesse ponto. Ao preencher os campos e clicar no botão "Adicionar Marcador", aparecerá um marcador no mapa nas referidas coordenadas. A figura da aba home acima mostra dois marcadores referentes aos locais do IME e do PCD que foram adicionados.


[...]

[...]

[...]

# Arquivos necessários:
Para cada região de planejamento é necessário baixar o Modelo Digital de Superfície (DSM) precisão 1 arcsec, o Modelo Digital de Terreno (DTM) precisão 1 arcsec e o Modelo Digital de Land Cover precisão 10 m. Verifique quais as coordenadas do ponto onde se deseja fazer a predição de enlace rádio e baixe os modelos de Elevação e Land Cover que abranjam esse local.

# Local de Download dos modelos digitais de Elevação e Land Cover:
O DSM pode ser baixado no link: [AW3D30 Global](https://opentopography.s3.sdsc.edu/minio/raster/AW3D30/AW3D30_global/) ou [OpenTopography](https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3).

O DTM pode ser baixado no link: [Bristol Data](https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn).

O Land Cover pode ser baixado no link: [WorldCover 2021](https://worldcover2021.esa.int/downloader).

Todos em formato .tif.

# Local onde os modelos digitais de Elevação e Land Cover devem ser colocados:

**DTM:**

Copie e cole o DTM dentro da pasta `Raster` que está na pasta do projeto com o nome padrão `S04W054.tif`, sendo 04 a latitude em graus do ponto mais ao sul do arquivo e 054 a longitude em graus do ponto mais a oeste do arquivo.

**DSM:**

Copie e cole o DSM dentro da pasta `dsm` que está na pasta do projeto com o nome padrão, por exemplo, `N05E047.tif`, sendo 05 a latitude em graus do ponto mais ao sul do arquivo e 057 a longitude em graus do ponto mais a oeste do arquivo.

**Land Cover:**

Copie e cole o arquivo de Land Cover dentro da pasta `LandCover` que está na pasta do projeto com o nome padrão, por exemplo, `N12W021.tif`, sendo 12 a latitude em graus do ponto mais ao sul do arquivo e 021 a longitude em graus do ponto mais a oeste do arquivo.

---
Link para download dos modelos digitas de superfície (pata dsm) e modelo digitas de terreno (pasta Raster) de Brasilia e Rio: Brasilia e Rio:
https://drive.google.com/drive/u/0/folders/1k5VBIpTVCEszTpykrOkx9SCyNPFD5Jhk

O land cover de Brasilia e Rio podes ser baixados no link: https://drive.google.com/drive/folders/1DBOuxq-hc90U32mBMR3DUIPvVci31MpK?usp=drive_link

Esse projeto pode ser baixado no botão "<>Code" acima. em seguida "Download .zip".

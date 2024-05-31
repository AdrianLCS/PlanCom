# ComPlanner

# Tema:

SOFTWARE DE PLANEJAMENTO DE MEIOS DE COMUNICAÇÕES COM BASE EM CARTAS DIGITALIZADAS DE UM AMBIENTE OPERACIONAL

# Conceito:

Sofrware que calcula a preda devido ao terreno, à vegetação e à fatores urbanos com base no modelo ITM, Ikegami e recomendações ITU. O usuário interage com um mapa, onde ele adiciona marcadores e equpamentos rádio e o sofrware retorna a preda de enlace e a informação se será ou não possível a comunicaçõa entre os dois pontos marcados.

# Função:

Cálculo da atenuação rádio.

# Motivação:

: A motivação de desenvolver um aplicativo de esteganografia partiu da importância do conhecimento de aspectos relacionados a comunicações para tomada de decisão em operações militares.

# Tutorial de execução do script:
  1- Instale o interpretador python e adicione python nas variáveis de ambiente. 

  2- No ambiente com python instalado, instale a biblioteca Rasterio:  "python -m pip install rasterio"
    
  3- No ambiente com python instalado, instale a biblioteca Pillow com o comando: "python -m pip install pillow"
  
  4- No ambiente com python instalado, instale a biblioteca Flask:  "python -m pip install flask"

  5- No ambiente com python instalado, instale a biblioteca Folium:  "python -m pip install folium"

  6- Baixe o toda a pasta do projeto programa e rode o arquivo App2.py: Abra o navegador e acessa a máquina em que o software está rodando pelo seu endreço IP na porta 5000.

# Tutorial de Uso:


A pagina inicial "Aba home"do app é demonstrada abaixo, nela o usuário pode visulizar os mapas, potos adicionos e camada de cobertura rádio:


![index](https://github.com/AdrianLCS/PlanCom/assets/114261968/6a4c2c03-113b-4422-bbed-cc1a9380c17c)

A Figura abaixo mostra a aba "Add Ponto", onde é realizada a adição de marcadores no mapa. Nessa aba, o usuário pode adicionar um ponto que vai aparecer no mapa através de um marcador. O usuário entra com o nome que deseja dar ao ponto, as coordenadas e a altura da antena, e seleciona o equipamento rádio que será operado nesse ponto. Ao preencher os campos e clicar no botão "Adicionar Marcador", aparecerá um marcador no mapa nas referidas coordenadas. A Figura da aba home acima mostra dois marcadores referentes aos locais do IME e do PCD que foram adicionados

![addpont](https://github.com/ProgramacaoAplicada2022/Adrian_Willian_Stegnographer/assets/114261968/c61e3f92-87f9-420d-b48d-02403a102ae4)


Para ocultar uma mensagem na Imagem o usuário deverá prencehre o campo senha e Mensagem, em seguida clicar no botão "Ocultar Mensagem na Imagem".
Ao clicar será gerada uma nova imagem no mesmo diretório da imagem selecionada, com o mesmo nome acrescido de "v2r" no formato PNG.

![Captura de tela_20221202_174627](https://user-images.githubusercontent.com/114261968/205383515-365073c5-3e66-4a4b-86bc-59770aa6ad0c.png)

Para revelar uma mensagem oculta em uma imagem, após a seleção de uma imagem que tem uma mensagem oculta, o usuário deverá preencher somente o campo Senha e clicar em "Revelar Mensagem da Imagem". 
Ao clicar a mensagem oculta aparecerá no campo mensagem.

![Captura de tela_20221202_175012](https://user-images.githubusercontent.com/114261968/205384044-01a1cc66-5b0c-4edc-aaaa-631591d74e78.png)


# Tutorial de compliação do código:
  1- Instale o interpretador python e adicione python nas variáveis de ambiente. 

  2- No ambiente com python instalado, instale a biblioteca wxPython:  "python -m pip install wxPython"
    
  3- No ambiente com python instalado, instale a biblioteca Pillow para manipulação de imagens com o comando: "python -m pip install pillow"

  4- No ambiente com python instalado, instale a biblioteca pyistaller para gerar o executável: "python -m pip install pyinstaller"

  5- Baixe o arquivo Steganographer.py e, no diretório do arquivo, execute o comando "pyinstaller --onefile --windowed Steganographer.py

o executável Steganographer.exe estará na pasta dist. para usa-lo basta clicar duas vezes e seguir o tutorial de uso aqui presente.
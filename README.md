# diabetes_project

# Descrição
Diabetes é uma doença causada pela produção insuficiente ou má absorção de insulina, hormônio que regula a glicose no sangue e garante energia para o organismo [1] .O projeto tem como objetivo desenvolver um modelo para classificar se o paciente terá ou não diabetes, com base em variáveis como por exemplo, obesidade, gênero, idade, polifagia e polidipsia. Um dicionário com o significado de cada variáveis está no disponível no documento `diabetes.rtf`.
# Como utilizar
No terminal bash,  clone o repositório da seguinte forma via url `git clone https://github.com/igorss77/diabetes_project.git` no seu diretório.
Em seguida, utilizando em um terminal com anaconda instalado, instale o environment do projeto com o seguinte comando `conda env create -f environment.yml`.
Ative o environment e execute `python main.py`para que o código do projeto seja executado.
Para acessar o notebook, digite `jupyter notebook` com a env instalada e abra a pasta `notebooks` e abra o notebook `diabetes_project.ipynb`.


# Estrutura do projeto

- data
|- diabetes_data_upload.csv  # base de dados

- figures  # princpais gráficos 
- docs
|- Diabetes.rtf  # dicionário de dados

- models
|- model.pkl  # modelo desenvolvido

- notebooks
|- diabetes_project.ipynb  # notenbook com experimentos  

- environment.yml # environment com as bibliotecas necessárias
- main.py # módulo principal de execução para treino do modelo
- data_preparation # módulo com funções necessárias ao código

# Performance do modelo

O modelo desenvolvido obteve foi capaz de acertar o diagnóstico em 92% dos pacientes avaliados. 

# Referências

[1][Saúde Brasil](http://antigo.saude.gov.br/saude-de-a-z/diabetes#:~:text=Diabetes%20%C3%A9%20uma%20doen%C3%A7a%20causada,das%20c%C3%A9lulas%20do%20nosso%20organismo.)  

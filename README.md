# Proposta C.2 - Previsão do número de infetados/mortes de doenças pulmonares

## Contexto

A criação de modelos de previsão sobre uma doença pode auxiliar uma melhor resposta no
combate à doença. Atualmente a pandemia COVID tem sobrecarregado os sistemas de saúde
consumindo a maioria dos seus recursos. Porém existem outras doenças pulmonares como a
pneumonia que continuam a provocar mortes.
Neste trabalho, pretende-se o desenvolvimento de uma ferramenta, com recurso a
métodos de deep learning e à linguagem Python, que seja capaz de prever o número de
infectados e mortos provocados por doenças pulmonares. A previsão será feita com base nos
dados fornecidos pelas entidades de saúde sobre o número de pessoas com doenças pulmonares
ao longo de um período de tempo.

A previsão do número de mortes será realizada utilizando Séries Temporais com diferentes registos de periodos temporais, nomeadamente: 

* registos diários
* registos semanais
* registos mensa


## Estrutura

O repositório está dividido em quatro pastas:

* [Resultados_Pesquisa](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Resultados_Pesquisa) : Onde é possível encontrar datasets pesquisados, bem como pesquisa acerca de como tratar os dados, bem como a criação
d modelos machine learning capazes de  aceitar Séries Temporais;  
* [Daily_Model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Daily_Model) : Encontram-se os datasets e ficheiros com o código Python criados para tratar e criar datasets, que posteriormente serão explorados e utilizados
para a concepção do modelo de machine learning capaz de fazer previsão diária de mortes; 
* [Weekly_Model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Weekly_model): Encontram-se os datasets ficheiros com o código Python criados para tratar e criar datasets, que posteriormente serão explorados e utilizados
para a concepção do modelo de machine learning capaz de fazer previsão semanal de mortes;
* [Monthly_Model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Monthly_Model): Encontram-se os datasets ficheiros com o código Python criados para tratar e criar datasets, que posteriormente serão explorados e utilizados
para a concepção do modelo de machine learning capaz de fazer previsão mensal de mortes.

## Dicionário dos dados

Para a concepção dos diferentes modelos de machine learning, foi necessário efectuar pesquisa acerca do que são Séries Temporais e quais os passos
necessário a efectuar no tratamentos dos dados, para que estes ficassem em condições de serem utilizados nos modelos.
No entanto, devido à falta de datasets que contivessem toda a possível informação necessária para uma boa previsão do modelo, foi necessário recolher dados de outro datasets
e assim construir um dataset mais robusto e com maior potencial de bom desempenho na previsão de resultados.
Nos subcapítulos seguintes, pormenorizamos mais sobre a proveniencia dos datasets bem como o tratamento que lhes foi aplicado para que fosse possível criar os datasets que irão mais tarde alimentar
os nossos modelos de previsão para Séries Temporais.

### Datasets diários

#### Origem dos dados  
... dizer a proveniencia dos datasets utilizados
##### Data1
...falar dicionario de dados ... enumerar as variáveis do dataset usado
##### Data2
...falar dicionario de dados
#### Tratamento dos dados
 ..... Dizer que no ficheiroXXXX se encontra com mais detalhe o tratamento feito ao datase
#### Datasets resultantes
 ..... dizer quais os datasets resultantes finais


### Datasets semanais
### Datasets mensais

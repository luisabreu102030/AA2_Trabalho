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
#### Tratamento dos dados
 ..... Dizer que no ficheiroXXXX se encontra com mais detalhe o tratamento feito ao datase
#### [daily_covid.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Daily_Model/daily_covid.csv)
...falar dicionario de dados ... enumerar as variáveis do dataset usado
#### [daily_diabetes.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Daily_Model/daily_diabetes.csv)
...falar dicionario de dados




### Datasets semanais

#### Origem dos dados  
[Dados de covid para Portugal](https://github.com/dssg-pt/covid19pt-data)

[Dados R(t) para Portugal](http://www.insa.min-saude.pt/category/areas-de-atuacao/epidemiologia/covid-19-curva-epidemica-e-parametros-de-transmissibilidade/)

[Dados Covid e outras doenças respiratórias para os Estados Unidos](https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Week-Ending-D/r8kw-7aab) 

[Dados climatéricos obtidos para Portugal e Estados Unidos](https://www.visualcrossing.com/weather/weather-data-services#/login)

[Dados de transporte/circulação nos Estados Unidos](https://www.bts.gov/covid-19/week-in-transportation)

[Dados de voos mundiais](https://www.flightradar24.com/data/statistics)

#### Tratamento dos dados
Para ver em mais detalhe o tratamento e construção dos datasets ver os notebooks:

[covid_portugal](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Weekly_model/covid_portugal.ipynb)

[usa_dataset](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Weekly_model/usa_dataset.ipynb)

#### Dados
Uma explicação do conteúdo em [covid_portugal.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Weekly_model/covid_portugal.csv).

Nome da coluna | Significado | Possíveis valores
------------ | ------------- | -------------
`data` | Data do primeiro dia da semana em questão | YYYY-MM-DD
`confirmados_arsnorte` | Novos Casos confirmados na ARS Norte      | Inteiro >= 0 |
`confirmados_arscentro` | Novos Casos confirmados na ARS Centro      | Inteiro >= 0 |
`confirmados_arslvt` | Novos Casos confirmados na ARS Lisboa e Vale do Tejo      | Inteiro >= 0 |
`confirmados_arsalentejo` | Novos Casos confirmados na ARS Alentejo     | Inteiro >= 0 |
`confirmados_arsalgarve` | Novos Casos confirmados na ARS Algarve    | Inteiro >= 0 |
`confirmados_acores` | Novos Casos confirmados na Região Autónoma dos Açores | Inteiro >= 0 |
`confirmados_madeira` | Novos Casos confirmados na Região Autónoma da Madeira  |  Inteiro >= 0 |
`confirmados_novos` | Número de novos casos confirmados comparativamente ao dia anterior | Inteiro >= 0 |
`recuperados` | Novos casos recuperados | Inteiro >= 0 
`obitos` | Número de óbitos | Inteiro >= 0
`internados` | Número de pacientes COVID-19 internados | Inteiro (caso o número seja negativo significa que o número de internados diminui relativamente à semana passada)
`internados_uci` | Número de pacientes COVID-19 internados em Unidades de Cuidados Intensivos | Inteiro (caso o número seja negativo significa que o número de internados em Unidades de Cuidados Intensivos diminui relativamente à semana passada)
`vigilancia` | Número de casos sob vigilância pelas autoridades de saúde | Inteiro (caso o número seja negativo significa que o número de casos sob vigilância diminui relativamente à semana passada)
`confirmados_0_9_f` | Número de novos casos confirmados do sexo feminino na faixa etária 0-9 anos | Inteiro >= 0
`confirmados_0_9_m` | Número de novos casos confirmados do sexo masculino na faixa etária 0-9 anos | Inteiro >= 0 
`confirmados_10_19_f` | Número de novos casos confirmados do sexo feminino na faixa etária 10-19 anos | Inteiro >= 0 
`confirmados_10_19_m` | Número de novos casos confirmados do sexo masculino na faixa etária 10-19 anos | Inteiro >= 0 
`confirmados_20_29_f` | Número de novos casos confirmados do sexo feminino na faixa etária 20-29 anos | Inteiro >= 0 
`confirmados_20_29_m` | Número de novos casos confirmados do sexo masculino na faixa etária 20-29 anos | Inteiro >= 0 
`confirmados_30_39_f` | Número de novos casos confirmados do sexo feminino na faixa etária 30-39 anos | Inteiro >= 0 
`confirmados_30_39_m` | Número de novos casos confirmados do sexo masculino na faixa etária 30-39 anos | Inteiro >= 0 
`confirmados_40_49_f` | Número de novos casos confirmados do sexo feminino na faixa etária 40-49 anos | Inteiro >= 0 
`confirmados_40_49_m` | Número de novos casos confirmados do sexo masculino na faixa etária 40-49 anos | Inteiro >= 0 
`confirmados_50_59_f` | Número de novos casos confirmados do sexo feminino na faixa etária 50-59 anos | Inteiro >= 0 
`confirmados_50_59_m` | Número de novos casos confirmados do sexo masculino na faixa etária 50-59 anos | Inteiro >= 0 
`confirmados_60_69_f` | Número de novos casos confirmados do sexo feminino na faixa etária 60-69 anos | Inteiro >= 0 
`confirmados_60_69_m` | Número de novos casos confirmados do sexo masculino na faixa etária 60-69 anos | Inteiro >= 0 
`confirmados_70_79_f` | Número de novos casos confirmados do sexo feminino na faixa etária 70-79 anos | Inteiro >= 0 
`confirmados_70_79_m` | Número de novos casos confirmados do sexo masculino na faixa etária 70-79 anos | Inteiro >= 0 
`confirmados_80_plus_f` | Número de novos casos confirmados do sexo feminino na faixa etária 80+ anos | Inteiro >= 0 
`confirmados_80_plus_m` | Número de novos casos confirmados do sexo masculino na faixa etária 80+ anos | Inteiro >= 0 
`confirmados_f` | Número de novos confirmados do sexo feminino | Inteiro >= 0 
`confirmados_m` | Número de novos confirmados do sexo masculino | Inteiro >= 0 
`obitos_arsnorte` | Novos Óbitos na ARS Norte      | Inteiro >= 0 
`obitos_arscentro` | Novos Óbitos na ARS Centro      | Inteiro >= 0 
`obitos_arslvt` | Novos Óbitos na ARS Lisboa e Vale do Tejo      | Inteiro >= 0 
`obitos_arsalentejo` | Novos Óbitos na ARS Alentejo     | Inteiro >= 0 
`obitos_arsalgarve` | Novos Óbitos na ARS Algarve    | Inteiro >= 0 
`obitos_acores` | Novos Óbitos na Região Autónoma dos Açores | Inteiro >= 0 
`obitos_madeira` | Novos Óbitos na Região Autónoma da Madeira  |  Inteiro >= 0 
`obitos_0_9_f` | Número de novos óbitos de pacientes do sexo feminino na faixa etária 0-9 anos | Inteiro >= 0
`obitos_0_9_m` | Número de novos óbitos de pacientes do sexo masculino na faixa etária 0-9 anos | Inteiro >= 0
`obitos_10_19_f` | Número de novos óbitos de pacientes do sexo feminino na faixa etária 10-19 anos | Inteiro >= 0 
`obitos_10_19_m` | Número de novos óbitos de pacientes do sexo masculino na faixa etária 10-19 anos | Inteiro >= 0 
`obitos_20_29_f` | Número de novos óbitos de pacientes do sexo feminino na faixa etária 20-29 anos | Inteiro >= 0 
`obitos_20_29_m` | Número de novos óbitos de pacientes do sexo masculino na faixa etária 20-29 anos | Inteiro >= 0 
`obitos_30_39_f` | Número de novos óbitos de pacientes do sexo feminino na faixa etária 30-39 anos | Inteiro >= 0 
`obitos_30_39_m` | Número de novos óbitos de pacientes do sexo masculino na faixa etária 30-39 anos | Inteiro >= 0 
`obitos_40_49_f` | Número de novos óbitos de pacientes do sexo feminino na faixa etária 40-49 anos | Inteiro >= 0 
`obitos_40_49_m` | Número de novos óbitos de pacientes do sexo masculino na faixa etária 40-49 anos | Inteiro >= 0
`obitos_50_59_f` | Número de novos óbitos de pacientes do sexo feminino na faixa etária 50-59 anos | Inteiro >= 0 
`obitos_50_59_m` | Número de novos óbitos de pacientes do sexo masculino na faixa etária 50-59 anos | Inteiro >= 0 
`obitos_60_69_f` | Número de novos óbitos de pacientes do sexo feminino na faixa etária 60-69 anos | Inteiro >= 0 
`obitos_60_69_m` | Número de novos óbitos de pacientes do sexo masculino na faixa etária 60-69 anos | Inteiro >= 0 
`obitos_70_79_f` | Número de novos óbitos de pacientes do sexo feminino na faixa etária 70-79 anos | Inteiro >= 0 
`obitos_70_79_m` | Número de novos óbitos de pacientes do sexo masculino na faixa etária 70-79 anos | Inteiro >= 0 
`obitos_80_plus_f` | Número de novos óbitos de pacientes do sexo feminino na faixa etária 80+ anos | Inteiro >= 0 
`obitos_80_plus_m` | Número de novos óbitos de pacientes do sexo masculino na faixa etária 80+ anos | Inteiro >= 0 
`obitos_f` | Número de novos óbitos de pacientes do sexo feminino | Inteiro >= 0 
`obitos_m` | Número de novos óbitos de pacientes do sexo masculino | Inteiro >= 0
`ativos` | Número de novos casos ativos | Inteiro (caso o número seja negativo significa que o número de casos ativos diminui relativamente à semana passada)
`internados_enfermaria` | Número de pacientes COVID-19 internados em Enfermaria (não Unidades de Cuidados Intensivos) | Inteiro (caso o número seja negativo significa que o número de internados na enfermaria diminui relativamente à semana passada)
`Rt_número_de_reprodução` | R(t) nacional | Float >= 0
`Max_Temp` | Média de temperaturas máxima registadas | Float
`Min_Temp` | Média de temperaturas mínima registada | Float
`Temperature`| Média de temperatura registada | Float
`Precipitation` | Média de precipitação registada | Float
`Wind_Speed` | Média da velocidade do vento registada | Float
`Wind_Direction` | Média da direção do vento registada | Float
`Visibility` | Média da visibilidade registada | Float
`Cloud_Cover` | Média da nebulosidade registada | Float
`Relative_Humidity` | Média da humidade registada | Float
`Rain` | Registo de chuva | Float
`Clear` | Registo de céu limpo | Float
`Partially_cloudy` | Registado de céu nublado | Float
`flights`	| Número médio de voos ocorridos em todo mundo | Float >= 0
`commercial_flights` | Número médio de voos comerciais ocorridos em todo mundo | Float >= 0

Uma explicação do conteúdo em [usa_dataset.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Weekly_model/usa_dataset.csv).

Nome da coluna | Significado | Possíveis valores
------------ | ------------- | -------------
`Start Date` | Data do primeiro dia da semana em questão | YYYY-MM-DD
`COVID-19 Deaths` | Número de mortes por COVID-19 | Inteiro >= 0
`Total Deaths` | Número Total de mortes | Inteiro >= 0
`Percent of Expected Deaths` | 
`Pneumonia Deaths` | Número de mortes por Pneumonia | Inteiro >= 0
`Pneumonia and COVID-19 Deaths` | Número de mortes por Pneumonia e COVID-19 | Inteiro >= 0
`Influenza Deaths` | Número de mortes por Influenza | Inteiro >= 0
`Pneumonia, Influenza, or COVID-19 Deaths` | Número de mortes por Pneumonia, COVID-19 e Influenza | Inteiro >= 0
`flights` | Número médio de voos ocorridos em todo mundo | Float >= 0
`commercial_flights` | Número médio de voos comerciais ocorridos em todo mundo | Float >= 0
`People Screened at Airports` | Número de pessoas examinadas nos aeroportos nos Estados Unidos | Inteiro >= 0
`US International Commercial Flights` | Número de voos comerciais internacionais ocorridos nos Estados Unidos | Inteiro >= 0
`Nr People staying home` | Número médio de pessoas que ficam em casa nos Estados Unidos | Float >= 0
`Nr People not staying home` |  Número médio de pessoas que não ficam em casa nos Estados Unidos | Float >= 0
`Nr of trips` | Número médio de viagens feitas nos Estados Unidos | Float >= 0
`Max_Temp` | Média de temperaturas máxima registadas | Float
`Min_Temp` | Média de temperaturas mínima registada | Float
`Temperature` | Média de temperatura registada | Float
`Precipitation` | Média de precipitação registada | Float
`Wind_Speed` | Média da velocidade do vento registada | Float
`Wind_Direction` | Média da direção do vento registada | Float
`Visibility` | Média da visibilidade registada | Float
`Cloud_Cover` | Média da nebulosidade registada | Float
`Relative_Humidity` | Média da humidade registada | Float
`Rain` | Registo de chuva | Float
`Clear` | Registo de céu limpo | Float
`Partially_cloudy` | Registado de céu nublado | Float


### Datasets mensais

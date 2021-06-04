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

O repositório está dividido em 6 pastas:

* [Datasets_utilizados](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Datasets_utilizados) : Onde é possível encontrar os datasets que compõem o dataset final que será utilizado na previsão do número de óbitos em Portugal;
* [Datasets_ignorados](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Datasets_ignorados) : Encontram-se os datasets pesquisados, e datasets já formados para outras doenças, no entanto que devido à falta de mais features foram por nós ignorados.
* [Tratamento_exploração](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Tratamento_Exploracao) : Encontra-se todo o processo de exploração de dados, bem como todo o seu tratamento no que diz respeito a missing values e à sua tranformação para datasets com registos de frequência diária, semanal e mensal.
* [Daily_Model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Daily_Model) : Encontram-se ficheiros com o código Python criados para criar os modelos machine learning, que posteriormente serão utilizados para prever o número de óbitos em Portugal. Estes modelos serão criados utilizandos Redes Recurrentes Neuronais LST e Redes Neuronais Convolucionais, e serão treinados utilizando Séries temporais cujos registos são diários.
para a concepção do modelo de machine learning capaz de fazer previsão diária de mortes; 
* [Weekly_Model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Weekly_model): Encontram-se ficheiros com o código Python criados para criar os modelos machine learning, que posteriormente serão utilizados para prever o número de óbitos em Portugal. Estes modelos serão criados utilizandos Redes Recurrentes Neuronais LST e Redes Neuronais Convolucionais, e serão treinados utilizando Séries temporais cujos registos são semanais.
* [Monthly_Model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Monthly_Model): Encontram-se ficheiros com o código Python criados para criar os modelos machine learning, que posteriormente serão utilizados para prever o número de óbitos em Portugal. Estes modelos serão criados utilizandos Redes Recurrentes Neuronais LST e Redes Neuronais Convolucionais, e serão treinados utilizando Séries temporais cujos registos são mensais.

## Dicionário dos dados

Para a concepção dos diferentes modelos de machine learning, foi necessário efectuar pesquisa acerca do que são Séries Temporais e quais os passos
necessário a efectuar no tratamentos dos dados, para que estes ficassem em condições de serem utilizados nos modelos.
No entanto, devido à falta de datasets que contivessem toda a possível informação necessária para uma boa previsão do modelo, foi necessário recolher dados de outro datasets
e assim construir um dataset mais robusto e com maior potencial de bom desempenho na previsão de resultados.
Nos subcapítulos seguintes, pormenorizamos mais sobre a proveniencia dos datasets bem como o tratamento que lhes foi aplicado para que fosse possível criar os datasets que irão mais tarde alimentar
os nossos modelos de previsão para Séries Temporais. O dataset criado contem registos cuja sua **frequencia é diária**

### Datasets utilizados

#### Origem dos dados 
 Aqui são referênciados os ados para a construção do dataset [covid_final.csv](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Tratamento_Exploracao/covid_final.csv) :


* [DSSG - Data Science for Social Good Portugal](https://github.com/dssg-pt/covid19pt-data/blob/master/data.csv) : Dados referentes ao COVID-19 em Portugal a partir de 26-02-2020;
* [Visualcrossing](https://www.visualcrossing.com) : Dados atmosféricos em Portugal a partir de 01-01-2020;
* [COVID-19 World Vaccination Progress](https://www.kaggle.com/gpreda/covid-world-vaccination-progress) : Total diário de vacinação do COVID-19 no mundo;
* [SNS - Serviço Nacional de Saúde Portugal](https://transparencia.sns.gov.pt/explore/dataset/acionamentos-de-meios-de-emergencia-medica/table/?sort=periodo&fbclid=IwAR1Q59_J2oyrap0gqnxUrqC2dKeSKev8seWiNqMPjsisxL4a_bgUMwKAgfE&refine.periodo=2021) : Evolução diária dos acionamentos de meios de emergência Médica;
* [Coronavirus Source Data from OurWorldInData.](https://ourworldindata.org/coronavirus-source-data) : Dados relativos ao COVID-19, provenientes de vários países do mundo.

#### Tratamento e exploração dos dados
Toda a exploração realizada sobre os dados e seu consequente tratamento aplicado aos datasets reunidos encontram-se, para uma mais fácil leitura em ficheiros Jupyter Notebook na pasta [Tratamento_Exploracao](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Tratamento_Exploracao).

#### Dataset final obtido : [covid_final.csv](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Tratamento_Exploracao/)

Nesta secção é apresentada a descrição dos dados presentes em covid_final.csv. Todas as features presentes neste dataset foram selecionadas recorrendo a técnicas de **feature selection** que se encontram em [feature_selection.ipymb](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Tratamento_Exploracao/feature_selection.ipymb). Previamente à escolha destes atributos, os dados sofreram reparações por forma a tratar todos os missing values e/ou timesteps em falta.

Nome da coluna|Significado|Valores possíveis
--------------|-----------|-----------------
Date|Registo diário | AAAA-MM--DD
`confirmados`| Casos confirmados de COVID-19|Inteiro >= 0
`confirmados_arsnorte`|Casos confirmados de COVID-19 na Autoridade Regional de Saúde Norte|Inteiro >= 0
`confirmados_arscentro`|Casos confirmados de COVID-19 na Autoridade Regional de Saúde Centro|Inteiro >= 0
`confirmados_arslvt`|Casos confirmados de COVID-19 na Autoridade Regional de Saúde Lisboa e Vale do Tejo|Inteiro >= 0
`confirmados_arsalentejo`|Casos confirmados de COVID-19 na Autoridade Regional de Saúde Alentejo|Inteiro >= 0
`confirmados_arsalgarve`|Casos confirmados de COVID-19 na Autoridade Regional de Saúde Algarve|Inteiro >= 0
`confirmados_acores`|Casos confirmados de COVID-19 na Autoridade Regional de Saúde Açores|Inteiro >= 0
`confirmados_madeira`|Casos confirmados de COVID-19 na Autoridade Regional de Saúde Madeira|Inteiro >= 0
`confirmados_novos`|Novos casos confirmados de COVID-19|Inteiro >= 0
`recuperados`|Número de pacientes recuperados|Inteiro >= 0
`obitos`|Número de óbitos|Inteiro >= 0
`internados_uci`| Número de internados em unidades de Cuidados Intensivos|Inteiro >= 0
`obitos_arsnorte`|Número de óbitos na Autoridade Regional de Saúde Norte|Inteiro >= 0
`obitos_arscentro`|Número de óbitos na Autoridade Regional de Saúde Centro|Inteiro >= 0
`obitos_arslvt`|Número de óbitos na Autoridade Regional de Saúde Lisboa e Vale do tejo|Inteiro >= 0
`obitos_arsalentejo`|Número de óbitos na Autoridade Regional de Saúde Alentejo|Inteiro >= 0
`obitos_arsalgarve`|Número de óbitos na Autoridade Regional de Saúde Algarve|Inteiro >= 0
`obitos_acores`|Número de óbitos na Autoridade Regional de Saúde Açores|Inteiro >= 0
`obitos_madeira`|Número de óbitos na Autoridade Regional de Saúde Madeira|Inteiro >= 0
`ativos|Número de casos ativos |Inteiro >= 0
`internados_enfermaria`|Número de pacientes COVID-19 internados em Enfermaria (não Unidades de Cuidados Intensivos)|Inteiro >= 0
`confirmados_0_9`|Número total de casos confirmados na faixa etária 0-9 anos|Inteiro >= 0
`confirmados_10_19`|Número total de casos confirmados na faixa etária 10-19 anos|Inteiro >= 0
`confirmados_20_29`|Número total de casos confirmados na faixa etária 20-29 anos|Inteiro >= 0
`confirmados_30_39`|Número total de casos confirmados na faixa etária 30-39 anos|Inteiro >= 0
`confirmados_40_49`|Número total de casos confirmados na faixa etária 40-49 anos|Inteiro >= 0
`confirmados_50_59`|Número total de casos confirmados na faixa etária 50-59 anos|Inteiro >= 0
`confirmados_60_69`|Número total de casos confirmados na faixa etária 60-69 anos|Inteiro >= 0
`confirmados_70_79`|Número total de casos confirmados na faixa etária 70-79 anos|Inteiro >= 0
`confirmados_80_plus`|Número total de casos confirmados na faixa etária 80 anos para cima|Inteiro >= 0
`obitos_0_9`|Número total de óbitos na faixa etária 0-9 anos|Inteiro >= 0
`obitos_10_19`|Número total de óbitos na faixa etária 10-19 anos|Inteiro >= 0
`obitos_20_29`|Número total de óbitos na faixa etária 20-29 anos|Inteiro >= 0
`obitos_30_39`|Número total de óbitos na faixa etária 30-39 anos|Inteiro >= 0
`obitos_40_49`|Número total de óbitos na faixa etária 40-49 anos|Inteiro >= 0
`obitos_50_59`|Número total de óbitos na faixa etária 50-59 anos|Inteiro >= 0
`obitos_60_69`|Número total de óbitos na faixa etária 60-69 anos|Inteiro >= 0
`obitos_70_79`|Número total de óbitos na faixa etária 70-79 anos|Inteiro >= 0
`obitos_80_plus`|Número total de óbitos na faixa etária 80 anos para cima|Inteiro >= 0
`Max_Temp`|Média de temperaturas máxima registadas| Float
`Min_Temp`|Média de temperaturas minimas registadas| Float
`Temperature`|Temperatura média (Max_Temp + Min_Temp)/2 | Float
`Precipitation`|Média de precipitação registada | Float
`Wind_Speed`|Média da velocidade do vento registada | Float
`Wind_Direction`|Média da direção do vento registada | Float
`Visibility`|Média da direção do vento registada | Float|
`Cloud_Cover`| Média da nebulosidade registada | Float
`Relative_Humidity`| Média da humidade registada | Float
`Rain`|Registo de chuva| 0 ou 1
`Clear`|Registo de céu limpo | 0 ou 1
`Partially_cloudy`|Registo de céu parcialmente nublado| 0 ou 1

**Nota :** Esta descrição de dataset é referente a um registo de dados com frequencia diária, no entanto para ser utilizado como dataset de frequencia de registo semanal e mensal algumas alterações foram necessárias realizar.
Para o dataset semanal, o seu registo de semana corresponde à data do primeiro dia da semana, no entanto o valor registado para essa semana é o somatório dos dias dessa semana. Semelhante alteração foi feita para obtermos um dataset com registos mensais, mas aqui a data de registo é referente ao último dia do mês e o seu valor é o somatório do valor do primeiro dia do mês até ao último dia do mês.


## Modelos deep learning

Neste projeto para a previsão do número de óbitos diários, semanais e mensais dois tipos de modelos de previsão:

* Recurrent Neural Network LSTM

 ![Tux, the Linux mascot](/assets/images/lstm.png)

* Convolutional neural Network

 ![Tux, the Linux mascot](/assets/images/lstm.png)

Todos os modelos testados, em conjunto com as suas otimizações experimentadas e resultados obtidos, encontram-se divididos em 3 pastas de ficheiros [Daily_Model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Daily_Model), [Weekly_model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Weekly_model) e [Monthly_Model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Monthly_Model).

#Hugo decide se quer manter ou apagar ou adicionar ao diario
### Datasets semanais AAAAAAAAAAAAAAAAAAAAAAAAAAAAA????????????????????????????????????
* Todos os datasets usados na construção dos datasets obtidos encontram-se em [Datasets](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Weekly_model/Datasets)
#### Origem dos dados  
* [Dados de covid para Portugal](https://github.com/dssg-pt/covid19pt-data)

* [Dados R(t) para Portugal](http://www.insa.min-saude.pt/category/areas-de-atuacao/epidemiologia/covid-19-curva-epidemica-e-parametros-de-transmissibilidade/)

* [Dados Covid e outras doenças respiratórias para os Estados Unidos](https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Week-Ending-D/r8kw-7aab) 

* [Dados climatéricos obtidos para Portugal e Estados Unidos](https://www.visualcrossing.com/weather/weather-data-services#/login)

* [Dados de transporte/circulação nos Estados Unidos](https://www.bts.gov/covid-19/week-in-transportation)

* [Dados de voos mundiais](https://www.flightradar24.com/data/statistics)

#### Tratamento dos dados
Para ver em mais detalhe o tratamento e construção dos datasets ver os notebooks:

* [covid_portugal](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Weekly_model/covid_portugal.ipynb)

* [usa_dataset](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Weekly_model/usa_dataset.ipynb)

#### Dados
* Uma explicação do conteúdo em [covid_portugal.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Weekly_model/covid_portugal.csv).

Nome da coluna | Significado | Possíveis valores
------------ | ------------- | -------------
`data` | Data do primeiro dia da semana em questão | YYYY-MM-DD

`internados_enfermaria` | Número de pacientes COVID-19 internados em Enfermaria (não Unidades de Cuidados Intensivos) | Inteiro (caso o número seja negativo significa que o número de internados na enfermaria diminui relativamente à semana passada)
`Rt_número_de_reprodução` | R(t) nacional | Float >= 0

`flights`	| Número médio de voos ocorridos em todo mundo | Float >= 0
`commercial_flights` | Número médio de voos comerciais ocorridos em todo mundo | Float >= 0

* Uma explicação do conteúdo em [usa_dataset.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Weekly_model/usa_dataset.csv).

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

*****SADJZKÇVN BZDKLFJBVN DFKJVBNOFNBBAEOÇFBN BAÇJRBN EOAI N''?????????????????????????????*****

# NOTA: [Datasets ignorados](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Datasets_ignorados)
Para além dos datasets acima referidos e documentados, foram criados ainda outros datasets que não foram utilizados no projeto devido à falta de dados que solidificacem o interesse nos mesmo. Como tal deixamos neste repositório esses mesmo datasets, com a esperança de poderem ajudar algum researcher.

### Datasets diários 
#### Origem dos dados 
##### Dados para a construção do dataset [daily_diabetes.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Daily_Model/daily_diabetes.csv) 
* [Office for national Statistics](https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/causesofdeath/adhocs/005259dailydeathoccurrencesbyallcausesanddiabetesmellitusicd10e10toe14englandandwales2012to2014) : Dados referentes a óbitos diários em Inglaterra de 01-01-2012 a 31-12-2014
* [ECA - Euopean Climate Assessmente & Dataset](https://www.ecad.eu/dailydata/customquery.php): Dados referentes ás temperaturas registadas em Inglaterra
* [Department for Environment Food & Rural Affairs - UK AIR](https://uk-air.defra.gov.uk/interactive-map): Dados referentes ao nível de Ozono registado em Inglaterra

#### Tratamento dos dados
Todo o tratamento de dados aplicado aos datasets reunidos encontram-se, para uma mais fácil leitura, no ficheiro Jupyter Notebook [process_data](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Daily_Model/process_data.ipynb), apesar de tamber se encontrarem em ficheiro Python [Process_Data]

#### Dados
#### [daily_diabetes.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Datasets_ignorados/Daily_datasets/daily_diabetes.csv)
Nome da coluna|Significado|Valores possíveis
--------------|-----------|-----------------
`Date`|Registo diário | AAAA-MM--DD
`All_Causes`|Número de óbitos por todas as causas|Inteiro >=0
`Diabetes`|Número de óbitos por diabetes| Inteiro >= 0
`Ozone`|Registo médio do nível de Ozono em Inglaterra| Float >=0
`Temperature`|temperatura média em Inglaterra| Float

### Datasets semanários 
#### Origem dos dados 
#### Tratamento dos dados
#### Dados

### Datasets mensais
#### Origem dos dados 

* [Dados de overdoses nos Estados Unidos da América](https://catalog.data.gov/dataset/vsrr-provisional-drug-overdose-death-counts) 
* * [Dados climatéricos obtidos para Estados Unidos da América](https://www.visualcrossing.com/weather/weather-data-services#/login)
#### Tratamento dos dados
Para ver em mais detalhe o tratamento e construção dos datasets ver o notebooks:

* [overdoses](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Datasets_ignorados/Montlhy_datasets/overdoses.ipynb)
#### Dados
* Uma explicação do conteúdo em [overdoses.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Datasets_ignorados/Montlhy_datasets/overdoses.csv).

Nome da coluna | Significado | Possíveis valores
------------ | ------------- | -------------
`Year` | Ano em questão | 2020
`Month` | Mês em questão | January
`Cocaine (T40.5)` | Número de mortes por overdose de Cocaína | Inteiro >= 0
`Heroin (T40.1)` | Número de mortes por overdose de Heroína | Inteiro >= 0 
`Methadone (T40.3)` | Número de mortes por overdose de Metadona | Inteiro >= 0
`Natural & semi-synthetic opioids (T40.2)` | Número de mortes por overdose de opióides naturais e semi-sintéticos | Inteiro >= 0
`Natural & semi-synthetic opioids, incl. methadone (T40.2, T40.3)` | Número de mortes por overdose de opióides naturais e semi-sintéticos, inculindo Metadona | Inteiro >= 0
`Natural, semi-synthetic, & synthetic opioids, incl. methadone (T40.2-T40.4)` | Número de mortes por overdose de opióides naturais, semi-sintéticos e sintéticos, inculindo Metadona | Inteiro >= 0
`Number of Drug Overdose Deaths` | Número total de mortes por overdose | Inteiro >= 0
`Opioids (T40.0-T40.4,T40.6)` | Número total de mortes por overdose por opióides | Inteiro >= 0
`Psychostimulants with abuse potential (T43.6)` | Número de mortes por overdose de psicoestimulantes com potencial de abuso | Inteiro >= 0
`Synthetic opioids, excl. methadone (T40.4)` | Número de mortes por overdose de opióides sintéticos, não incluíndo Metadona | Inteiro >= 0



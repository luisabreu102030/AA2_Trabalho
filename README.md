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
* [Tratamento_exploração](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Tratamento_Exploracao) : Encontra-se todo o processo de exploração de dados, bem como todo o seu tratamento no que diz respeito a missing values e à sua tranformação para datasets com registos de frequência diária, semanal e mensal. Esta pasta encontra-se dividida em duas pastas, uma contém o tratamento dos dados outra a sua exploração bem como a seleção de atributos. 
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
* [Visualcrossing](https://www.visualcrossing.com) : Dados atmosféricos em Portugal e Estados Unidos da América;
* [COVID-19 World Vaccination Progress](https://www.kaggle.com/gpreda/covid-world-vaccination-progress) : Total diário de vacinação do COVID-19 no mundo;
* [SNS - Serviço Nacional de Saúde Portugal](https://transparencia.sns.gov.pt/explore/dataset/acionamentos-de-meios-de-emergencia-medica/table/?sort=periodo&fbclid=IwAR1Q59_J2oyrap0gqnxUrqC2dKeSKev8seWiNqMPjsisxL4a_bgUMwKAgfE&refine.periodo=2021) : Evolução diária dos acionamentos de meios de emergência Médica;
* [Coronavirus Source Data from OurWorldInData.](https://ourworldindata.org/coronavirus-source-data) : Dados relativos ao COVID-19, provenientes de vários países do mundo;
* [Dados Covid e outras doenças respiratórias para os Estados Unidos](https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Week-Ending-D/r8kw-7aab) : Dados relativos ao COVID-19 e outras doenças respiratórias nos Estados unidos da América;
* [Dados de transporte/circulação nos Estados Unidos](https://www.bts.gov/covid-19/week-in-transportation) : Dados referentes á circulação de pessoas e transportes nos Estados unidos da América;
* [Dados de voos mundiais](https://www.flightradar24.com/data/statistics) : Dados de vôos realizados nos Estados unidos da América.

#### Tratamento e exploração dos dados
Toda a exploração realizada sobre os dados e seu consequente tratamento aplicado aos datasets reunidos encontram-se, para uma mais fácil leitura em ficheiros Jupyter Notebook na pasta [Tratamento_Exploracao](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Tratamento_Exploracao).

#### Dataset final obtido : [covid_final.csv](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Tratamento_Exploracao/)

Nesta secção é apresentada a descrição dos dados presentes em covid_final.csv. Todas as features presentes neste dataset foram selecionadas recorrendo a técnicas de **feature selection** que se encontram em [feature_selection.ipymb](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Tratamento_Exploracao/feature_selection.ipymb). Previamente à escolha destes atributos, os dados sofreram reparações por forma a tratar todos os missing values e/ou timesteps em falta. **A variável objetivo é a variável obito**.

Nome da coluna|Significado|Valores possíveis
--------------|-----------|-----------------
Date|Registo diário | AAAA-MM--DD
`confirmados_novos`| Número de novos casos confirmados em Portugal | Inteiro >= 0
`recuperados`|Número de novos pacientes recuperados em Portugal | Inteiro >= 0
`Temperature`|Temperatura média em Portugal | Float
`Visibility`|Grau de visibilidadae médio em Portugal |Float
`total_testes`|Total de testes ao COVID-19 realizados em Portugal | Inteiro >= 0
`testes_pcr`|Númerero de testes PCR realizados em Portugal | Inteiro >= 0
`testes_antigenio`|Númerero de testes Antigénio realizados em Portugal | Inteiro >= 0
`new_cases_per_million_Austria`| Número de novos casos por milhão de pessoas na Austria | Float >= 0
`new_deaths_per_million_Austria`| Número de mortes por Covid por milhão de pessoas na Austria | Float >= 0
`new_cases_per_million_Belgium`| Número de novos casos por milhão de pessoas na Bélgica | Float >= 0
`new_deaths_per_million_Belgium`| Número de mortes por Covid por milhão de pessoas na Bélgica | Float >= 0
`icu_patients_per_million_Belgium`| Número de pacientes em unidades de cuidados intensivos por milhão de pessoas na Bélgica | Float >= 0
`hosp_patients_per_million_Belgium`| Número de pacientes hospitalizados por milhão de pessoas na Bélgica | Float >= 0
`new_deaths_per_million_Bulgaria`| Número de mortes por Covid por milhão de pessoas na Bulgária | Float >= 0
`new_cases_per_million_Canada`| Número de novos casos por milhão de pessoas no Canadá | Float >= 0
`new_deaths_per_million_Canada`| Número de mortes por Covid por milhão de pessoas no Canadá | Float >= 0 
`icu_patients_per_million_Canada`| Número de pacientes em unidades de cuidados intensivos por milhão de pessoas no Canadá | Float >= 0
`new_cases_per_million_Cyprus`| Número de novos casos por milhão de pessoas no Chipre | Float >= 0
`new_deaths_per_million_Czechia`| Número de mortes por Covid por milhão de pessoas na República Checa | Float >= 0
`hosp_patients_per_million_Czechia`|Número de pacientes hospitalizados por milhão de pessoas na República Checa | Float >= 0 
`new_cases_per_million_Denmark`|Número de novos casos por milhão de pessoas na Dinamarca | Float >= 0
`new_tests_per_thousand_Estonia`|Número de novos testes por mil de pessoas na Estónia | Float >= 0 
`icu_patients_per_million_France`|Número de pacientes em unidades de cuidados intensivos por milhão de pessoas na França | Float >= 0
`new_cases_per_million_Georgia`|Número de novos casos por  milhão de pessoas na Geórgia | Float >= 0
`new_cases_per_million_Germany`|Número de novos casos por  milhão de pessoas na Alemanha | Float >= 0
`new_deaths_per_million_Germany`|Número de mortes por Covid por  milhão de pessoas na Alemanha | Float >= 0
`icu_patients_per_million_Germany`|Número de pacientes em unidades de cuidados intensivos por milhão de pessoas na Alemanha | Float >= 0
`new_deaths_per_million_Hungary`|Número de mortes por Covid por  milhão de pessoas na Hungria | Float >= 0
`hosp_patients_per_million_Hungary`|Número de pacientes hospitalizados por  milhão de pessoas na Hungria | Float >= 0
`positive_rate_India`|Taxa de testes positivos à COVID-19 na Índia | Float >= 0 
`new_cases_per_million_Ireland`|Número de novos casos por  milhão de pessoas na Irlanda | Float >= 0
`new_deaths_per_million_Ireland`|Número de mortes por Covid por  milhão de pessoas na Irlanda | Float >= 0
`icu_patients_per_million_Ireland`|Número de pacientes em unidades de cuidados intensivos por milhão de pessoas na Irlanda | Float >= 0
`hosp_patients_per_million_Ireland`|Número de pacientes hospitalizados por milhão de pessoas na Irlanda | Float >= 0 
`new_tests_per_thousand_Ireland`|Número de novos testes por mil de pessoas na Iralanda | Float >= 0
`positive_rate_Ireland`|Taxa de testes positivos à COVID-19 na Irlanda | Float >= 0 
`new_tests_per_thousand_Italy`|Número de novos testes por mil de pessoas na Itália | Float >= 0
`new_cases_per_million_Japan`|Número de novos casos por  milhão de pessoas no Japão | Float >= 0
`new_deaths_per_million_Japan`|Número de mortes por Covid por  milhão de pessoas no Japão | Float >= 0
`new_cases_per_million_Latvia`|Número de novos casos por  milhão de pessoas na Letônia | Float >= 0
`new_deaths_per_million_Latvia`|Número de mortes por Covid por  milhão de pessoas na Letônia |Float >= 0
`hosp_patients_per_million_Latvia`|Número de pacientes hospitalizados por milhão de pessoas na Letônia | Float >= 0 
`new_tests_per_thousand_Latvia`|Número de novos testes por mil de pessoas na Letônia | Float >= 0
`new_cases_per_million_Lithuania`|Número de novos casos por milhão de pessoas na Lituânia | Float >= 0
`new_tests_per_thousand_Lithuania`|Número de novos testes por mil de pessoas na Lituânia |Float >= 0
`icu_patients_per_million_Luxembourg`|Número de pacientes em unidades de cuidados intensivos por milhão de pessoas no Luxemburgo | Float >= 0
`new_deaths_per_million_Malta`|Número de mortes por Covid por  milhão de pessoas em Malta | Float >= 0
`new_tests_per_thousand_Malta`|Número de novos testes por mil de pessoas em Malta | Float >= 0
`positive_rate_Malta`|Taxa de testes positivos à COVID-19 em Malta |Float >= 0
`new_cases_per_million_Monaco`|Número de novos casos por  milhão de pessoas no Mónaco | Float >= 0
`new_cases_per_million_Montenegro`|Número de novos casos por  milhão de pessoas em Montenegro | Float >= 0
`positive_rate_Mozambique`|Taxa de testes positivos à COVID-19 em Moçambique | Float >= 0 
`new_deaths_per_million_Poland`|Número de mortes por Covid por  milhão de pessoas na Polónia | Float >= 0
`positive_rate_Romania`|Taxa de testes positivos à COVID-19 na Roménia | Float >= 0 
`new_cases_per_million_Russia`|Número de novos casos por  milhão de pessoas na Rússia | Float >= 0
`new_deaths_per_million_Russia`|Número de mortes por Covid por  milhão de pessoas na Rússia | Float >= 0 
`new_deaths_per_million_Serbia`|Número de mortes por Covid por milhão de pessoas na Sérvia | Float >= 0
`positive_rate_Serbia`|Taxa de testes positivos à COVID-19 na Sérvia | Float >= 0 
`new_cases_per_million_Slovakia`|Número de novos casos por milhão de pessoas na Eslováquia | Float >= 0
`new_tests_per_thousand_Slovakia`|Número de novos testes por mil de pessoas na Eslováquia | Float >= 0
`new_cases_per_million_Slovenia`|Número de novos casos por milhão de pessoas na Eslovénia | Float >= 0
`new_deaths_per_million_Slovenia`|Número de mortes por Covid por milhão de pessoas na Eslovénia | Float >= 0
`icu_patients_per_million_Slovenia`|Número de pacientes em unidades de cuidados intensivos por milhão de pessoas na Eslovénia | Float >= 0
`new_tests_per_thousand_Slovenia`|Número de novos testes por mil pessoas na Eslovénia |Float >= 0 
`new_deaths_per_million_Switzerland`|Número de mortes por Covid por milhão de pessoas na |Float >= 0
`new_cases_per_million_Turkey`|Número de novos casos por milhão de pessoas na Turquia |Float >= 0 
`positive_rate_Ukraine`|Taxa de testes positivos à COVID-19 na Ucránia | Float >= 0
`new_deaths_per_million_United Arab Emirates`|Número de mortes por Covid por milhão de pessoas nos Emirados Árabes Unidos | Float >= 0
`tests_per_case_United Arab Emirates`|Número de novos casos por milhão de pessoas nos Emirados Árabes Unidos | Float >= 0
`new_cases_per_million_United Kingdom`|Número de novos casos por milhão de pessoas no Reino Unido | Float >= 0
`new_deaths_per_million_United Kingdom`|Número de mortes por Covid por milhão de pessoas na Reino Unido | Float >= 0
`icu_patients_per_million_United Kingdom`|Número de pacientes em unidades de cuidados intensivos por milhão de pessoas na Reino Unido | Float >= 0
`hosp_patients_per_million_United Kingdom`|Número de pacientes hospitalizados por milhão de pessoas na Reino Unido | Float >= 0
`positive_rate_United Kingdom`|Taxa de testes positivos à COVID-19 no Reino Unido | Float >= 0
`new_cases_per_million_United States`|Número de novos casos por milhão de pessoas nos Estados Unidos da América | Float >= 0
`new_deaths_per_million_United States`|Número de mortos por milhão de pessoas nos Estados Unidos da América |Float >= 0
`obitos`|Número de mortos por COVID-19 em Portugal |Inteiro >= 0
 
**Nota 1 :** Esta descrição de dataset é referente a um registo de dados com frequencia diária, no entanto para ser utilizado como dataset de frequencia de registo semanal e mensal algumas alterações foram necessárias realizar.
Para o dataset semanal, o seu registo de semana corresponde à data do primeiro dia da semana, no entanto o valor registado para essa semana é o somatório dos dias dessa semana. Semelhante alteração foi feita para obtermos um dataset com registos mensais, mas aqui a data de registo é referente ao último dia do mês e o seu valor é o somatório do valor do primeiro dia do mês até ao último dia do mês.

**Nota 2 :** No dataset que foi adaptado para registos semanais foram acrescentados as seguintes variáveis independentes:

Nome da coluna | Significado | Possíveis valores
------------ | ------------- | -------------
`Start Date` | Data do primeiro dia da semana em questão | YYYY-MM-DD
`Pneumonia Deaths` | Número de mortes por Pneumonia nos Estados Unidos da América | Inteiro >= 0 
`Pneumonia and COVID-19` | Número de mortes por COVID-19 nos Estados Unidos da América | Inteiro >= 0
`Influenza Deaths` | Número de mortes por Influenza nos Estados Unidos da América | Inteiro >= 0  
`flights` | Número médio de voos ocorridos em todo mundo | Float >= 0 
`commercial_flights` | Número médio de voos comerciais ocorridos em todo mundo | Float >= 0
`People Screened at Airports` | Número de pessoas examinadas nos aeroportos nos Estados Unidos | Inteiro >= 0  
`US International Commercial Flights` | Número de voos comerciais internacionais ocorridos nos Estados Unidos | Inteiro >= 0  
`Nr People not staying home` |  Número médio de pessoas que não ficam em casa nos Estados Unidos | Float >= 0  
`Nr of trips` | Número médio de viagens feitas nos Estados Unidos | Float >= 0  


## Modelos deep learning

Neste projeto para a previsão do número de óbitos diários, semanais e mensais dois tipos de modelos de previsão:

* Recurrent Neural Network LSTM

 ![Tux, the Linux mascot](/assets/images/lstm.png)

* Convolutional neural Network

 ![Tux, the Linux mascot](/assets/images/lstm.png)

Todos os modelos testados, em conjunto com as suas otimizações experimentadas e resultados obtidos, encontram-se divididos em 3 pastas de ficheiros [Daily_Model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Daily_Model), [Weekly_model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Weekly_model) e [Monthly_Model](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Monthly_Model).

## Relatório

No ficheiro [relatorio.pdf](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/relatorio.pdf) é possível encontrar a discussão dos resultados obtidos entre outras considerações sobre o projeto.


# NOTA: [Datasets ignorados](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Datasets_ignorados)
Para além dos datasets acima referidos e documentados, foram criados ainda outros datasets que não foram utilizados no projeto devido à falta de dados que solidificacem o interesse nos mesmo. Como tal deixamos neste repositório esses mesmo datasets, com a esperança de poderem ajudar algum researcher.

### Datasets diários 
#### Origem dos dados 
##### Dados para a construção do dataset [daily_diabetes.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Daily_Model/daily_diabetes.csv) 
* [Office for national Statistics](https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/causesofdeath/adhocs/005259dailydeathoccurrencesbyallcausesanddiabetesmellitusicd10e10toe14englandandwales2012to2014) : Dados referentes a óbitos diários em Inglaterra de 01-01-2012 a 31-12-2014
* [ECA - Euopean Climate Assessmente & Dataset](https://www.ecad.eu/dailydata/customquery.php): Dados referentes ás temperaturas registadas em Inglaterra
* [Department for Environment Food & Rural Affairs - UK AIR](https://uk-air.defra.gov.uk/interactive-map): Dados referentes ao nível de Ozono registado em Inglaterra

#### Tratamento dos dados
Todo o tratamento de dados aplicado aos datasets reunidos encontram-se, para uma mais fácil leitura, no ficheiro Jupyter Notebook [process_data](https://github.com/luisabreu102030/AA2_Trabalho/tree/main/Tratamento_Exploracao/tratamento_data), apesar de tamber se encontrarem em ficheiro Python [Process_Data]

#### Dados
#### [daily_diabetes.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Datasets_ignorados/Daily_datasets/daily_diabetes.csv)
Nome da coluna|Significado|Valores possíveis
--------------|-----------|-----------------
`Date`|Registo diário | AAAA-MM--DD
`All_Causes`|Número de óbitos por todas as causas|Inteiro >=0
`Diabetes`|Número de óbitos por diabetes| Inteiro >= 0
`Ozone`|Registo médio do nível de Ozono em Inglaterra| Float >=0
`Temperature`|temperatura média em Inglaterra| Float


### Datasets mensais
#### Origem dos dados 

* [Dados de overdoses nos Estados Unidos da América](https://catalog.data.gov/dataset/vsrr-provisional-drug-overdose-death-counts) 
* [Dados climatéricos obtidos para Estados Unidos da América](https://www.visualcrossing.com/weather/weather-data-services#/login)
#### Tratamento dos dados
Para ver em mais detalhe o tratamento e construção dos datasets ver o notebook [overdoses](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Datasets_ignorados/Montlhy_datasets/overdoses.ipynb)
#### Dados
* Uma explicação sobre o conteúdo de [overdoses.csv](https://github.com/luisabreu102030/AA2_Trabalho/blob/main/Datasets_ignorados/Montlhy_datasets/overdoses.csv).

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



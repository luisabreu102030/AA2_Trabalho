import numpy as np
import pandas as pd
from datetime import datetime


#Load dataset
import requests
import json

#####################################################
#                  Load datasets                    #
#####################################################
def load_normal_dataset(path):
    return pd.read_csv(path)

def load_datasets(path):
    return pd.read_csv(path, sep=';')


#####################################################
#              Datasets to csv file                 #
#####################################################

def to_csv_file(df, file_name):
    df.to_csv(file_name)


#####################################################
#                  Prepare datasets                 #
#####################################################

#####################################################
#               Diabetes Inglaterra                 #
#####################################################
#Prepare Diabetes_2014_England
def prepare_data_diabetes(df_raw):

    df_aux = df_raw.copy()

    df_aux.columns = ['Date','All_Causes','Diabetes']
    #print(df_aux)
    
    #tirar o espaço da string All_Causes
    df_aux['All_Causes'] = df_aux['All_Causes'].str.replace(" ",'')

    # alterar tipo das colunas All_Causes, Diabetes para numerico :
    # Alterar tipo da coluna Date para datetime
    df_aux["Diabetes"] = pd.to_numeric(df_aux["Diabetes"])
    df_aux['All_Causes'] = pd.to_numeric(df_aux['All_Causes'])

    #Ver se tem duplicados
    r  = df_aux.duplicated(subset='Date').sum()
    #print(r)

    #Ver se tem missing values
    mv = df_aux.isnull().sum()
    #print(mv)

    #somar mortes no mesmo dia
    #df_aux = df_aux.groupby(['Date']).sum()

    pd.set_option('display.max_rows', None)
    #print(df_aux)
    return df_aux

#Prepare mean_temp_london
def prepare_data_temperature_england(df_raw,name):

    df_aux = df_raw.copy()
    #print(df_aux.dtypes)
    #print (df_aux.head())
    #pd.set_option('display.max_rows', None)
    #print(df_aux)
    #unique values of Station id (só tem 1 estaçao)
    #print (len(df_aux['STAID'].unique()))
    #unique values od date (não tem duplicados)
    #print(len(df_aux['DATE'].unique()))

    #drop colunas STAID,Q_TG
    df_aux = df_aux.drop(columns=['SOUID','STAID','Q_TG'])

    #tornar valores da temperatura, em valores reais   temperatura*0.1
    df_aux['TG'] = df_aux['TG']*0.1

    #converter integer date para a data correta: 19600101 para 1960-01-01
    df_aux['DATE'] = pd.to_datetime(df_aux['DATE'], format='%Y%m%d')

    #modificar data de 1960-01-01 para 01/01/1960
    df_aux["DATE"] = pd.to_datetime(df_aux["DATE"]).dt.strftime('%d/%m/%Y')

    nome ='temp_'+name
    #renomear colunas
    df_aux.columns = ['Date', nome]
    #print(df_aux.head())
    #print(df_aux.dtypes)
    return df_aux


def prepare_data_ozone_england(df_raw,name):
    df_aux = df_raw.copy()

    #Ver se tem missing values
    #mv = df_aux.isnull().sum()
    #print('Estacao: ',name,' nº: ',mv)
    #print(df_aux.columns)
    #substituir missing values pelo valor médio da hora
    mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8,mean9,mean10,mean11,mean12 = df_aux['01:00'].mean(),df_aux['02:00'].mean(),df_aux['03:00'].mean(),df_aux['04:00'].mean(),df_aux['05:00'].mean(), df_aux['06:00'].mean(),df_aux['07:00'].mean(),df_aux['08:00'].mean(),df_aux['09:00'].mean(),df_aux['10:00'].mean(),df_aux['11:00'].mean(),df_aux['12:00'].mean()
    mean13, mean14, mean15, mean16, mean17, mean18, mean19, mean20, mean21, mean22, mean23, mean24 =  df_aux['13:00'].mean(),df_aux['14:00'].mean(),df_aux['15:00'].mean(),df_aux['16:00'].mean(),df_aux['17:00'].mean(),df_aux['18:00'].mean(),df_aux['19:00'].mean(),df_aux['20:00'].mean(),df_aux['21:00'].mean(),df_aux['22:00'].mean(),df_aux['23:00'].mean(),df_aux['24:00'].mean()


    df_aux['01:00'] = df_aux['01:00'].fillna(mean1)
    df_aux['02:00'] = df_aux['02:00'].fillna(mean2)
    df_aux['03:00'] = df_aux['03:00'].fillna(mean3)
    df_aux['04:00'] = df_aux['04:00'].fillna(mean4)
    df_aux['05:00'] = df_aux['05:00'].fillna(mean5)
    df_aux['06:00'] = df_aux['06:00'].fillna(mean6)
    df_aux['07:00'] = df_aux['07:00'].fillna(mean7)
    df_aux['08:00'] = df_aux['08:00'].fillna(mean8)
    df_aux['09:00'] = df_aux['09:00'].fillna(mean9)
    df_aux['10:00'] = df_aux['10:00'].fillna(mean10)
    df_aux['11:00'] = df_aux['11:00'].fillna(mean11)
    df_aux['12:00'] = df_aux['12:00'].fillna(mean12)
    df_aux['13:00'] = df_aux['13:00'].fillna(mean13)
    df_aux['14:00'] = df_aux['14:00'].fillna(mean14)
    df_aux['15:00'] = df_aux['15:00'].fillna(mean15)
    df_aux['16:00'] = df_aux['16:00'].fillna(mean16)
    df_aux['17:00'] = df_aux['17:00'].fillna(mean17)
    df_aux['18:00'] = df_aux['18:00'].fillna(mean18)
    df_aux['19:00'] = df_aux['19:00'].fillna(mean19)
    df_aux['20:00'] = df_aux['20:00'].fillna(mean20)
    df_aux['21:00'] = df_aux['21:00'].fillna(mean21)
    df_aux['22:00'] = df_aux['22:00'].fillna(mean22)
    df_aux['23:00'] = df_aux['23:00'].fillna(mean23)
    df_aux['24:00'] = df_aux['24:00'].fillna(mean24)

    # Ver se tem missing values
    #mv = df_aux.isnull().sum()
    #print('Estacao: ', name, ' nº: ', mv)

    #coluna media diária de ozono
    nome = 'ozone'+ name
    df_aux[nome] = df_aux.iloc[:, 1:25].mean(axis=1)

    #drop colunas das horas
    cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    df_aux = df_aux.drop(df_aux.columns[cols], axis=1)

    #modificar data de 1960-01-01 para 01/01/1960
    df_aux["Date"] = pd.to_datetime(df_aux["Date"]).dt.strftime('%d/%m/%Y')

    return df_aux


def unifie_temp_mean(f_data_temp_bognor_regis,f_data_temp_blackpool,f_data_temp_durham,f_data_temp_nottingham):
    df_diabetes = pd.merge(f_data_temp_bognor_regis,f_data_temp_blackpool, on='Date', how='inner')
    df_diabetes = pd.merge(df_diabetes, f_data_temp_durham, on='Date', how='inner')
    df_diabetes = pd.merge(df_diabetes, f_data_temp_nottingham, on='Date', how='inner')
    #drop das colunas do ozone das cidades
    df_diabetes['Temperature'] = df_diabetes.iloc[:, 1:4].mean(axis=1)
    cols= [1,2,3,4]
    df_diabetes = df_diabetes.drop(df_diabetes.columns[cols], axis=1)
    #print(df_diabetes.shape)
    #print(df_diabetes.head())
    return df_diabetes

def unifie_ozone_mean(f_data_ozone_hull,f_data_ozone_norwich,f_data_ozone_wirral,f_data_ozone_london):
    df_ozone = pd.merge(f_data_ozone_hull,f_data_ozone_norwich, on='Date', how='inner')
    df_ozone = pd.merge(df_ozone,f_data_ozone_wirral, on='Date', how='inner')
    df_ozone = pd.merge(df_ozone, f_data_ozone_london, on='Date', how='inner')
    #print(df_ozone.shape)
    #drop das colunas do ozone das cidades
    df_ozone['Ozone'] = df_ozone.iloc[:, 1:4].mean(axis=1)
    cols= [1,2,3,4]
    df_ozone = df_ozone.drop(df_ozone.columns[cols], axis=1)
    #print(df_ozone.shape)
    return df_ozone


def unifie_diabetes_datasets(f_data_diabetes, f_data_temp, f_data_ozone):

    df_diabetes = pd.merge(f_data_diabetes,f_data_temp, on='Date', how='inner')
    df_diabetes = pd.merge(df_diabetes, f_data_ozone, on='Date', how='inner')
    #pd.set_option('display.max_rows', None)
    #print(df_diabetes)

    return df_diabetes





#####################################################
#                  Covid_19 Portugal                #
#####################################################
#prepare covid data
def prepare_data_covid(df_raw_covid):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df_aux = df_raw_covid.copy()
    #print(df_aux.shape)

    #print("Number of missing Values: ")
    #print(df_aux.isnull().sum())


    df_aux = df_raw_covid.drop(columns=['data_dados', 'confirmados_estrangeiro', 'lab', 'suspeitos', 'vigilancia',
                                        'n_confirmados', 'cadeias_transmissao', 'transmissao_importada',
                                        'sintomas_tosse', 'sintomas_febre', 'sintomas_cefaleia',
                                        'sintomas_dores_musculares', 'sintomas_dificuldade_respiratoria', 'sintomas_fraqueza_generalizada',
                                        'obitos_estrangeiro', 'recuperados_arsnorte','recuperados_arscentro',
                                        'recuperados_arslvt','recuperados_arsalentejo',
                                        'recuperados_arsalgarve','recuperados_acores', 'recuperados_madeira',
                                        'recuperados_estrangeiro','confirmados_desconhecidos_m','confirmados_desconhecidos_f',
                                        'incidencia_nacional','incidencia_continente','rt_nacional', 'rt_continente',
                                        'obitos_f', 'obitos_m','confirmados_desconhecidos','confirmados_f','confirmados_m',
                                        'internados'], inplace=False)


    #somar colunas casos confirmados com mesma faixa etaria ignorando sexo
    df_aux['confirmados_0_9'] = df_aux.apply(lambda x: x['confirmados_0_9_f'] + x['confirmados_0_9_m'], axis=1)
    df_aux['confirmados_10_19'] = df_aux.apply(lambda x: x['confirmados_10_19_f'] + x['confirmados_10_19_m'], axis=1)
    df_aux['confirmados_20_29'] = df_aux.apply(lambda x: x['confirmados_20_29_f'] + x['confirmados_20_29_m'], axis=1)
    df_aux['confirmados_30_39'] = df_aux.apply(lambda x: x['confirmados_30_39_f'] + x['confirmados_30_39_m'], axis=1)
    df_aux['confirmados_40_49'] = df_aux.apply(lambda x: x['confirmados_40_49_f'] + x['confirmados_40_49_m'], axis=1)
    df_aux['confirmados_50_59'] = df_aux.apply(lambda x: x['confirmados_50_59_f'] + x['confirmados_50_59_m'], axis=1)
    df_aux['confirmados_60_69'] = df_aux.apply(lambda x: x['confirmados_60_69_f'] + x['confirmados_60_69_m'], axis=1)
    df_aux['confirmados_70_79'] = df_aux.apply(lambda x: x['confirmados_70_79_f'] + x['confirmados_70_79_m'], axis=1)
    df_aux['confirmados_80_plus'] = df_aux.apply(lambda x: x['confirmados_80_plus_f'] + x['confirmados_80_plus_m'], axis=1)

    df_aux = df_aux.drop(columns=['confirmados_0_9_f','confirmados_0_9_m',
                            'confirmados_10_19_f','confirmados_10_19_m',
                            'confirmados_20_29_f','confirmados_20_29_m',
                            'confirmados_30_39_f','confirmados_30_39_m',
                            'confirmados_40_49_f','confirmados_40_49_m',
                            'confirmados_50_59_f','confirmados_50_59_m',
                            'confirmados_60_69_f','confirmados_60_69_m',
                            'confirmados_70_79_f','confirmados_70_79_m',
                            'confirmados_80_plus_f','confirmados_80_plus_m'], inplace=False)


    #somar colunas obitos confirmados com mesma faixa etaria ignorando sexo
    df_aux['obitos_0_9'] = df_aux.apply(lambda x: x['obitos_0_9_f'] + x['obitos_0_9_m'], axis=1)
    df_aux['obitos_10_19'] = df_aux.apply(lambda x: x['obitos_10_19_f'] + x['obitos_10_19_m'], axis=1)
    df_aux['obitos_20_29'] = df_aux.apply(lambda x: x['obitos_20_29_f'] + x['obitos_20_29_m'], axis=1)
    df_aux['obitos_30_39'] = df_aux.apply(lambda x: x['obitos_30_39_f'] + x['obitos_30_39_m'], axis=1)
    df_aux['obitos_40_49'] = df_aux.apply(lambda x: x['obitos_40_49_f'] + x['obitos_40_49_m'], axis=1)
    df_aux['obitos_50_59'] = df_aux.apply(lambda x: x['obitos_50_59_f'] + x['obitos_50_59_m'], axis=1)
    df_aux['obitos_60_69'] = df_aux.apply(lambda x: x['obitos_60_69_f'] + x['obitos_60_69_m'], axis=1)
    df_aux['obitos_70_79'] = df_aux.apply(lambda x: x['obitos_70_79_f'] + x['obitos_70_79_m'], axis=1)
    df_aux['obitos_80_plus'] = df_aux.apply(lambda x: x['obitos_80_plus_f'] + x['obitos_80_plus_m'], axis=1)

    df_aux = df_aux.drop(columns=['obitos_0_9_f','obitos_0_9_m',
                            'obitos_10_19_f','obitos_10_19_m',
                            'obitos_20_29_f','obitos_20_29_m',
                            'obitos_30_39_f','obitos_30_39_m',
                            'obitos_40_49_f','obitos_40_49_m',
                            'obitos_50_59_f','obitos_50_59_m',
                            'obitos_60_69_f','obitos_60_69_m',
                            'obitos_70_79_f','obitos_70_79_m',
                            'obitos_80_plus_f','obitos_80_plus_m'], inplace=False)

    #drop das ultimas 2 rows do dataset
    df_aux = df_aux.drop(df_aux.tail(2).index)

    #fill row com missing values aplicando ffill method, n vejo problema pois aind a não haviam casos e faltando um dia mantem o valor do dia anterio
    #obitos_80_plus
    df_aux = df_aux.fillna(method='ffill', limit=1)
    #obitos_arsalentejo
    df_aux = df_aux.fillna(method='ffill', limit=19)
    #como os nan estão ao inicio da pandemia, é natural n terem valor, logo vou substituir esses NaN por 0
    df_aux = df_aux.fillna(0)

    #print(df_aux.isnull().sum())
    #print(df_aux)
    #print("Number of missing Values: ")
    #print(df_aux.isnull().sum())
    # counting the duplicates
    #dups = df_aux.pivot_table(index=['data','confirmados'], aggfunc='size')
    # displaying the duplicate Series
    #print(dups)
    #Não tem duplicados
    #linhas e colunas do dataset:
    #print(df_aux.shape)


    df_aux.columns = ['Date', 'confirmados', 'confirmados_arsnorte', 'confirmados_arscentro',
     'confirmados_arslvt', 'confirmados_arsalentejo',
     'confirmados_arsalgarve', 'confirmados_acores', 'confirmados_madeira',
     'confirmados_novos', 'recuperados', 'obitos', 'internados_uci',
     'obitos_arsnorte', 'obitos_arscentro', 'obitos_arslvt',
     'obitos_arsalentejo', 'obitos_arsalgarve', 'obitos_acores',
     'obitos_madeira', 'ativos', 'internados_enfermaria', 'confirmados_0_9',
     'confirmados_10_19', 'confirmados_20_29', 'confirmados_30_39',
     'confirmados_40_49', 'confirmados_50_59', 'confirmados_60_69',
     'confirmados_70_79', 'confirmados_80_plus', 'obitos_0_9',
     'obitos_10_19', 'obitos_20_29', 'obitos_30_39', 'obitos_40_49',
     'obitos_50_59', 'obitos_60_69', 'obitos_70_79', 'obitos_80_plus']


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    #print(df_aux.columns)
    #print(df_aux)

    return df_aux


#append temperatures portugal datasets
def append_temp_dataframes(df_raw_temp_2020_01, df_raw_temp_2020_02, df_raw_temp_2020_03, df_raw_temp_2020_04,
                           df_raw_temp_2020_05, df_raw_temp_2020_06):
    df_raw = df_raw_temp_2020_01.append(df_raw_temp_2020_02, ignore_index = True)
    df_raw = df_raw.append(df_raw_temp_2020_03, ignore_index = True)
    df_raw = df_raw.append(df_raw_temp_2020_04, ignore_index = True)
    df_raw = df_raw.append(df_raw_temp_2020_05, ignore_index = True)
    df_raw = df_raw.append(df_raw_temp_2020_06, ignore_index = True)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    #print(df_raw)

    return df_raw



#Prepare temperatures médias Portugal

#One hot encoding
def condition_ohe_rain(condition):
    res =0
    if(condition == 'Rain' or condition == 'Rain, Partially cloudy'):
        res=1

    return res


def condition_ohe_clear(condition):
    res =0
    if(condition == 'Clear'):
        res=1

    return res


def condition_ohe_cloudy(condition):
    res =0
    if(condition == 'Rain, Partially cloudy' or condition == 'Partially cloudy'):
        res=1

    return res


def prepare_data_temp(df_raw):
    #print(df_raw.head)
    df_aux = df_raw.copy()
    #Ver se tem missing values
    mv = df_aux.isnull().sum()
    #print(mv)


    #drop de colunas com muitos missing values e a coluna nome do país, colunas referentes á neve(têm apenas valores ='s a 0)
    df_aux = df_aux.drop(columns=['Name','Wind Chill','Heat Index', 'Wind Gust', 'Snow', 'Snow Depth'])
    #print(df_aux.shape)

    #rename columns
    df_aux.columns = ['Date', 'Max_Temp','Min_Temp','Temperature','Precipitation','Wind_Speed','Wind_Direction','Visibility','Cloud_Cover','Relative_Humidity','Conditions']

    #one hot encoding weather conditions
    df_aux['Rain'] = np.nan
    df_aux['Clear'] = np.nan
    df_aux['Partially_cloudy'] = np.nan
    
    df_aux['Rain'] = df_aux.apply(lambda x: condition_ohe_rain(x['Conditions']), axis=1)
    df_aux['Clear'] = df_aux.apply(lambda x: condition_ohe_clear(x['Conditions']), axis=1)
    df_aux['Partially_cloudy'] = df_aux.apply(lambda x: condition_ohe_cloudy(x['Conditions']), axis=1)

    df_aux = df_aux.drop(columns='Conditions')

    df_aux["Date"] = pd.to_datetime(df_aux["Date"]).dt.strftime('%d-%m-%Y')


    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    #print(df_aux)
    #print(df_aux.shape)
    #print(df_aux['Conditions'].unique())

    return df_aux



def unifie_covid_datasets(df_data_covid, df_data_temp):

    df_covid = pd.merge(df_data_covid,df_data_temp, on='Date', how='left')

    #print(df_data_covid.shape)
    #print(df_data_temp.shape)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    #print(df_data_temp)
    #print(df_covid)
    #print(df_covid.shape)

    #mv = df_covid.isnull().sum()
    #print(mv)

    return df_covid


#####################################################
#                 Funções finais                    #
#####################################################

def to_csv_diabetes():
    ################## Diabetes Datasets #################################
    df_raw_diabetes = load_datasets("daily_datasets/Diabetes_2014_England.csv")
    df_raw_temp_bognor_regis = load_normal_dataset("daily_datasets/mean_temp_bognor_regis.csv")
    df_raw_temp_blackpool = load_normal_dataset("daily_datasets/mean_temp_blackpool.csv")
    df_raw_temp_durham = load_normal_dataset("daily_datasets/mean_temp_durham.csv")
    df_raw_temp_nottingham = load_normal_dataset("daily_datasets/mean_temp_nottingham.csv")
    df_raw_ozone_hull = load_normal_dataset("daily_datasets/hull_freetown_ozone.csv")
    df_raw_ozone_norwich = load_normal_dataset("daily_datasets/norwich_lakenfields_ozone.csv")
    df_raw_ozone_wirral = load_normal_dataset("daily_datasets/wirral_tranmere_ozone.csv")
    df_raw_ozone_london = load_normal_dataset("daily_datasets/london_ozone.csv")


    # preparacao inicial de datasets diabetes
    f_data_diabetes = prepare_data_diabetes(df_raw_diabetes)

    # preparacao inicial de datasets temperaturas
    f_data_temp_bognor_regis = prepare_data_temperature_england(df_raw_temp_bognor_regis, 'BOGNOR')
    f_data_temp_blackpool = prepare_data_temperature_england(df_raw_temp_blackpool, 'BLACKP')
    f_data_temp_durham = prepare_data_temperature_england(df_raw_temp_durham, 'DURHAM')
    f_data_temp_nottingham = prepare_data_temperature_england(df_raw_temp_nottingham, 'NOTTI')

    # fazer média temperatura em inglaterra
    df_temp_england = unifie_temp_mean(f_data_temp_bognor_regis, f_data_temp_blackpool, f_data_temp_durham,
                                       f_data_temp_nottingham)

    # preparacao inicial de datasets ozono
    f_data_ozone_hull = prepare_data_ozone_england(df_raw_ozone_hull, 'HULL')
    f_data_ozone_norwich = prepare_data_ozone_england(df_raw_ozone_norwich, 'NORWICH')
    f_data_ozone_wirral = prepare_data_ozone_england(df_raw_ozone_wirral, 'WINRRAL')
    f_data_ozone_london = prepare_data_ozone_england(df_raw_ozone_london, 'LONDON')

    # fazer média ozono em inglaterra
    df_ozone_england = unifie_ozone_mean(f_data_ozone_hull, f_data_ozone_norwich, f_data_ozone_wirral,
                                         f_data_ozone_london)

    # join de todos os dados para formar o dataset diabetes:
    df_diabetes = unifie_diabetes_datasets(f_data_diabetes, df_ozone_england, df_temp_england)

    #print(df_diabetes)
    df_diabetes["Date"] = pd.to_datetime(df_diabetes["Date"])
    df_diabetes = df_diabetes.set_index('Date')
    print(df_diabetes.head())
    # datasets para csv file
    to_csv_file(df_diabetes, "daily_diabetes.csv")



def to_csv_covid():
    ########################## Covid Dataset ###########################
    #load datasets
    df_raw_covid = load_normal_dataset("daily_datasets/daily_covid_19_portugal.csv")
    df_raw_temp_2020_01 = load_normal_dataset('daily_datasets/temp_portugal_01012020_31032020.csv')
    df_raw_temp_2020_02 = load_normal_dataset('daily_datasets/temp_portugal_01042020_30062020.csv')
    df_raw_temp_2020_03 = load_normal_dataset('daily_datasets/temp_portugal_01072020_30092020.csv')
    df_raw_temp_2020_04 = load_normal_dataset('daily_datasets/temp_portugal_01102020_31122020.csv')
    df_raw_temp_2020_05 = load_normal_dataset('daily_datasets/temp_portugal_01012021_31032021.csv')
    df_raw_temp_2020_06 = load_normal_dataset('daily_datasets/temp_portugal_042021.csv')
    #Prepare dataset covid
    df_data_covid = prepare_data_covid(df_raw_covid)
    #append datasets temperaturas
    df_raw_appended_temp = append_temp_dataframes(df_raw_temp_2020_01,df_raw_temp_2020_02,df_raw_temp_2020_03,df_raw_temp_2020_04,df_raw_temp_2020_05,df_raw_temp_2020_06)
    #Prepare dataset temperaturas médias
    df_data_temp = prepare_data_temp(df_raw_appended_temp)
    #Unifie all datasets
    df_covid = unifie_covid_datasets(df_data_covid, df_data_temp)
    #print(df_covid.shape)


    #print(df_data_covid.columns)
    df_covid["Date"] = pd.to_datetime(df_covid["Date"])
    df_covid = df_covid.set_index('Date')
    # datasets para csv file
    to_csv_file(df_covid, "daily_covid.csv")




#####################################################
#                        Main                       #
#####################################################

if __name__ == '__main__':

    to_csv_diabetes()
    to_csv_covid()








'''

#teste de call ás api, que não foi necessário utilizar
#make_api_call_test()
#make_api_meteomatics()



Muitos missing values
def  prepare_sugar_price(df_raw_sugar):
    df_aux = df_raw_sugar.copy()
    #renomear colunas
    df_aux.columns = ['Date','Sugar']
    # modificar data de 1960-01-01 para 01/01/1960
    df_aux["Date"] = pd.to_datetime(df_aux["Date"]).dt.strftime('%d/%m/%Y')

    #print(df_aux.head())
    return df_aux

def make_api_call_test():
    response = requests.get('https://api.github.com/events')
    print(response.status_code)
    print(response.json())

    #colocar resposta em json num ficheiro
    with open('data.json', 'w') as outfile:
        json.dump(response.json(), outfile)

    #converter json em csv
        #orient posibilidades:
        #‘split‘: dict like {index -> [index], columns ->, data -> [values]}
        #‘records‘: list like [{column -> value}, … , {column -> value}]
        #‘index‘: dict like {index -> {column -> value}}
        #‘columns‘: dict like {column -> {index -> value}}
        #‘values‘: just the values array

    pdObj = pd.read_json('data.json', orient='records')
    print(pdObj)
    #funciona


def make_api_meteomatics():
    #lisboa: Coordenadas geográficas Lisboa, Latitude: 38.7071, Longitude: -9.13549
    response = requests.get('https://api.ipma.pt/open-data/observation/climate/temperature-max/lisboa/mtxmn-1106-lisboa.csv')
    print(response.status_code)
'''

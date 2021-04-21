import numpy as np
import pandas as pd

#Load dataset
def load_datasets(path, path1):
    return pd.read_csv(path, sep=';'), pd.read_csv(path1, sep=';')

def prepare_data_diabetes(df_raw):
    df_aux = df_raw.drop(columns=['All causes (A00-R99)'], inplace=False)

    df_aux = df_aux
    df_aux.columns = ['Date', 'Diabetes']


    # alterar tipo da coluna Diabetes para numerico :
    # Alterar tipo da coluna Date para datetime
    df_aux["Diabetes"] = pd.to_numeric(df_aux["Diabetes"])
    df_aux["Date"] = pd.to_datetime(df_aux["Date"])

    #somar mortes no mesmo dia
    df_aux = df_aux.groupby(['Date']).sum()

    pd.set_option('display.max_rows', None)
    #print(df_aux)
    return df_aux


def prepare_data_ebola(df_raw_ebola):
    print(df_raw_ebola.columns)
    df_aux = df_raw_ebola.drop(columns=['Country', 'Localite', 'Sources', 'Link'], inplace=False)
    
    #Selecionar só registos de mortes
    are_deaths_cases = df_aux['Category'] == 'Deaths'
    df_aux = df_aux[are_deaths_cases]
    #Drop da coluna Category
    df_aux = df_aux.drop(columns=['Category'], inplace=False)
    df_aux.columns = ['Ebola', 'Date']


    #remover espaços da coluna Ebola
    df_aux["Ebola"] = df_aux['Ebola'].str.replace(' ', '')

    #alterar tipo da coluna Ebola para numerico :
    #Alterar tipo da coluna Date para datetime
    df_aux["Ebola"] = pd.to_numeric(df_aux["Ebola"])
    df_aux["Date"] = pd.to_datetime(df_aux["Date"])

    #somar mortes no mesmo dia
    df_aux = df_aux.groupby(['Date']).sum()

    #ordenar data
    #df_aux['Date'] = pd.to_datetime(df_aux['Date'])
    #df_aux = df_aux.sort_values(by='Date')


    pd.set_option('display.max_rows', None)
    #print(df_aux)
    return df_aux


def unite_dataframe(df_data_diabetes, df_data_ebola):
    #Diabetes tem valores \to\do o ano, ebola já não ....
    df_aux = pd.merge(df_data_diabetes, df_data_ebola, how="left", on=["Date", "Date"])

    pd.set_option('display.max_rows', None)
    print(df_aux)
    return df_aux



if __name__ == '__main__':
    
    #load datasets
    df_raw_diabetes, df_raw_ebola = load_datasets("Diabetes_2014_England.csv", "daily_ebola_deaths_africa_2014.csv")
    
    #preparacao inicial de datasets
    df_data_diabetes = prepare_data_diabetes(df_raw_diabetes)
    df_data_ebola = prepare_data_ebola(df_raw_ebola)
    
    #join dos datasets
    df_total = unite_dataframe(df_data_diabetes, df_data_ebola)

    #DUVIDA: Como tratar os NaN. Que periodo de dias escolher ?
    
    
import numpy as np
import pandas as pd


#Load dataset
def load_dataset(path):
    return pd.read_csv(path)


if __name__ == '__main__':

    df = load_dataset("daily_dataset.csv")
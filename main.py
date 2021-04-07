import numpy as np
import pandas as pd



def processData():
    d = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'col2': [10, 11, 12, 13, 14, 15, 16, 17, 18]}
    dataset = pd.DataFrame(data=d)


    print(dataset.head(5))


if __name__ == '__main__':
    print("Hello world")
    processData()
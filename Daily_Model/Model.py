import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


#for replicability purposes
tf.random.set_seed(91195003)
np.random.seed(91190530)
#for an easy reset backend session state
tf.keras.backend.clear_session()


if __name__ == '__main__':
    print("Hello world")

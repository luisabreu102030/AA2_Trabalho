import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# for replicability purposes
tf.random.set_seed(91195003)
np.random.seed(91195003)

# for an easy reset backend session state
tf.keras.backend.clear_session()


# load dataset
def load_dataset(path=r'C:\Users\Veloso\Desktop\MEI\2Semestre\CSC\TP2\Yahoo Stock Price\yahoo_stock.csv'):
    return pd.read_csv(path)


# split data into training and validation sets
def split_data(training, perc=10):
    train_idx = np.arange(0, int(len(training) * (100 - perc) / 100))
    val_idx = np.arange(int(len(training) * (100 - perc) / 100 + 1), len(training))
    return train_idx, val_idx


# preparing the data for the LSTM
def prepare_data(df):
    # Get data from 2015 to the end of 2018
    df_raw = df[:1134]
    # Get pair to evaluate for time series
    # We are going to predict the opening value for each day
    df_raw = df_raw.drop(columns=['High', 'Low', 'Volume', 'Adj Close'])
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values("Date")
    df_raw = df_raw.set_index("Date")
    return df_raw


def data_normalization(df, norm_range=(-1, 1)):
    # [-1, 1] for LSTM due to the internal use of tanh by the memory cell
    scaler = MinMaxScaler(feature_range=norm_range)
    # df[['cases']] = scaler.fit_transform(df[['cases']])
    df[['Open']] = scaler.fit_transform(df[['Open']])
    df[['Close']] = scaler.fit_transform(df[['Close']])
    return scaler


# Plot time series data
def plot_confirmed_cases(data):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(data)), data)
    plt.title('Opening Value of S&P500 Index')
    plt.ylabel('Value')
    plt.xlabel('Days')
    plt.show()


# plot learning curve
def plot_learning_curves(history, epochs):
    # accuracies and losses
    # dict_keys(['loss', 'mae', 'rmse', 'val_loss', 'val_mae', 'val_rmse'])
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    rmse = history.history['rmse']
    val_rmse = history.history['val_rmse']
    epochs_range = range(epochs)
    # creating figure
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.plot(epochs_range, mae, label='Training MAE')
    plt.plot(epochs_range, val_mae, label='Validation MAE')
    plt.plot(epochs_range, rmse, label='Training RMSE')
    plt.plot(epochs_range, val_rmse, label='Validation RMSE')
    plt.legend(loc='upper right')
    plt.title('Training/Validation Loss')
    plt.show()


# Preparing the dataset for the CNN
def to_supervised(df, timesteps):
    data = df.values
    X, y = list(), list()
    # iterate over the training set to create X and y
    dataset_size = len(data)
    for curr_pos in range(dataset_size):
        # end of the input sequence is the current position + the number of the timesteps of the input sequence
        input_index = curr_pos + timesteps
        # end of the labels corresponds to the end of the input sequence + 1
        label_index = input_index + 1
        # if we have enough data for this sequence
        if label_index < dataset_size:
            X.append(data[curr_pos:input_index, :])
            y.append(data[input_index:label_index, 0])
    # using np.flot32 for GPU performance
    return np.array(X).astype('float32'), np.array(y).astype('float32')


# Building the model
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


def build_model(timesteps, features, filters=16, kernel_size=5, pool_size=2):
    # using the functional API
    inputs = tf.keras.layers.Input(shape=(timesteps, features))
    # microarchitecture
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                               data_format='channels_last')(inputs)
    x = tf.keras.layers.AveragePooling1D(pool_size=pool_size, data_format='channels_first')(x)
    # last layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(filters)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    # the model
    cnnModel = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_model')
    tf.keras.utils.plot_model(cnnModel, 'S&P500_cnnmodel.png', show_shapes=True)
    return cnnModel


def compile_and_fit(model, epochs, batch_size):
    # compile
    model.compile(loss=rmse, optimizer=tf.keras.optimizers.Adam(), metrics=['mae', rmse])
    # fit
    hist_list = list()
    loss_list = list()
    # Time Series Cross Validator
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    for train_index, test_index in tscv.split(X):
        train_idx, val_idx = split_data(train_index, perc=10)  # further split into training and validation sets
        # build data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_index], y[test_index]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                            shuffle=False)
        metrics = model.evaluate(X_test, y_test)
        hist_list.append(history)
        loss_list.append(metrics[2])
    plot_learning_curves(hist_list, 'history')
    plot_learning_curves(loss_list, 'loss')
    return model, hist_list, loss_list


def forecast(model, df, timesteps, multisteps, scaler):
    input_seq = df[-timesteps:].values  # getting the last sequence of known value
    inp = input_seq
    forecasts = list()
    print(inp)
    # multistep tells us how many iterations we want to perform
    # i. e. how many days we want to predict
    for step in range(1, multisteps + 1):
        # implement
        inp = inp.reshape(1, timesteps, 1)
        pred = model.predict(inp)
        yhat_inversed = scaler.inverse_transform(pred)
        forecasts.append(yhat_inversed[0][0])
        # prepare new imput to forecast the next day
        inp = np.append(inp[0], pred)
        inp = inp[-timesteps:]
        inp = np.append(inp[0], [[pred[0][0], pred[0][1]]], axis=0)  # adiciona previsão recente ao input
        inp = inp[-timesteps:]  # vai ao input buscar os ultimos timesteps registados
        print(step)

    return forecasts


def plot_forecast(data, forecasts):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(data)), data, color='green', label='Confirmed')
    plt.plot(range(len(data) - 1, len(data) + len(forecasts) - 1), forecasts, color='red', label='Forecasts')

    plt.title('Predicted Pice of the S&P500')
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend()
    plt.show()


# Main Execution
timesteps = 5  # number of days that make a sequence
univariate = 1  # number of features used by the model
multisteps = 50  # number of days to forecast - we will forecast the next 5 days
cv_splits = 3  # time series validator
epochs = 25
batch_size = 7  # 7 sequences of 5 days - which corresponds to a window of 7 days in a batch

# the dataframes

df_raw = load_dataset()
df_data = prepare_data(df_raw)
df = df_data.copy()
plot_confirmed_cases(df_data)  # the plot you saw previously
scaler = data_normalization(df)  # scaling data to [-1,1]

# our supervised problem
X, y = to_supervised(df, timesteps)
print("Training shape: ", X.shape)
print("Training labels shape:", y.shape)
# fitting the model
# model = build_model(timesteps, univariate)
# model, hist_list, loss_list = compile_and_fit(model, epochs, batch_size)

# Now that we have “tuned” our model, we should retrain it with all the available data
# (as we did in SBS) and obtain real predictions for tomorrow, the day after, …
# We want to forecast the next five days after today!
# We just need to:

model = build_model(timesteps, univariate)
model.compile(loss=rmse, optimizer=tf.keras.optimizers.Adam(), metrics=['mae', rmse])
model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=False)
model.save(r'C:\Users\Veloso\Desktop\MEI\2Semestre\CSC\TP2\Yahoo Stock Price\savedModels')

# Recursive multi-step forecast!!!
forecasts = forecast(model, df, timesteps, multisteps, scaler)
plot_forecast(df_data, forecasts)
from loguru import logger
import streamlit as st
import datetime as dt
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dropout, BatchNormalization, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from keras.utils import Progbar
from keras.callbacks import LambdaCallback
import optuna
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math


########## LSTM ##########
def lstm_algorithm(ticker, df, best_params):
    forecast_out = 5
    epochs = best_params['epochs']
    batch_size = best_params['batch_size']
    unit_number = best_params['unit_number']
    dropout_rate = best_params['dropout_rate']
    l2_strength = best_params['l2_strength']
    activation = best_params['activation']
    lr_factor = best_params['lr_factor']
    min_lr = best_params['min_lr']
    optimizer = best_params['optimizer']

    # Use all columns except 'Close' for training
    features = df.columns.difference(['Close'])
    target = 'Close'
    
    dataset_train = df.iloc[0:int(0.8*len(df)), :]
    dataset_test = df.iloc[int(0.8*len(df)):, :]
    training_set = dataset_train[features].values
    target_set = dataset_train[[target]].values

    sc = MinMaxScaler(feature_range=(0, 1))
    sc_target = MinMaxScaler(feature_range=(0, 1))

    training_set_scaled = sc.fit_transform(training_set)
    target_set_scaled = sc_target.fit_transform(target_set)

    X_train = []
    y_train = []
    num_features = training_set_scaled.shape[1]

    for i in range(forecast_out, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-forecast_out:i, :])
        y_train.append(target_set_scaled[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))

    # Model architecture
    model = Sequential()
    model.add(LSTM(unit_number, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=l2(l2_strength), activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(unit_number, return_sequences=True, kernel_regularizer=l2(l2_strength), activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(unit_number, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=10, min_lr=min_lr)

    # Adding Progress bar for epochs
    st_container = st.empty()

    # Add progress bar to the container
    st_progress_bar = st_container.progress(0)

    def update_progress_bar(epoch, logs):
        st_progress_bar.progress((epoch+1)/epochs)

    lambda_callback = LambdaCallback(on_epoch_end=update_progress_bar)
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, 
                callbacks=[early_stopping, reduce_lr, lambda_callback])

    # Prediction on test set
    real_stock_price = dataset_test[[target]].values
    testing_set = sc.transform(dataset_test[features].values)
    X_test = []
    
    for i in range(forecast_out, len(testing_set)):
        X_test.append(testing_set[i-forecast_out:i, :])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_features))
    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc_target.inverse_transform(predicted_stock_price)
    error_lstm = math.sqrt(mean_squared_error(real_stock_price[forecast_out:], predicted_stock_price))

    # Recursive multi-step forecast
    forecast_set = []
    last_sequence = X_train[-1, :, :].reshape(1, X_train.shape[1], num_features)

    for _ in range(forecast_out):
        next_pred = model.predict(last_sequence)
        forecast_set.append(float(next_pred[0, 0].item()))
        
        next_pred_reshaped = np.zeros((1, last_sequence.shape[2]))
        next_pred_reshaped[0, 0] = next_pred[0, 0]
        
        new_sequence = np.concatenate((last_sequence[0, 1:, :], next_pred_reshaped), axis=0)
        
        last_sequence = new_sequence.reshape(1, last_sequence.shape[1], last_sequence.shape[2])

    # Inverse scaling
    forecast_set = sc_target.inverse_transform(np.array(forecast_set).reshape(-1, 1))
    
    # Calculate mean of forecast
    lstm_pred = int(forecast_set[0, 0])
    mean_forecast = int(np.mean(forecast_set))
    lstm_forecast_set = forecast_set.flatten()
    
    st_container.empty()
    
    return lstm_pred, lstm_forecast_set, mean_forecast, error_lstm, real_stock_price[forecast_out:], predicted_stock_price


# def lstm_algorithm(ticker, df, best_params):
#     forecast_out = 5
#     epochs = best_params['epochs']
#     batch_size = best_params['batch_size']
#     unit_number = best_params['unit_number']
#     dropout_rate = best_params['dropout_rate']
#     l2_strength = best_params['l2_strength']
#     activation = best_params['activation']
#     lr_factor = best_params['lr_factor']
#     min_lr = best_params['min_lr']
#     optimizer = best_params['optimizer']

#     # Use all columns except 'Close' for training
#     features = df.columns.difference(['Close'])
#     target = 'Close'
    
#     dataset_train = df.iloc[0:int(0.8*len(df)), :]
#     dataset_test = df.iloc[int(0.8*len(df)):, :]
#     training_set = dataset_train[features].values
#     target_set = dataset_train[[target]].values

#     sc = MinMaxScaler(feature_range=(0, 1))
#     sc_target = MinMaxScaler(feature_range=(0, 1))

#     training_set_scaled = sc.fit_transform(training_set)
#     target_set_scaled = sc_target.fit_transform(target_set)

#     X_train = []
#     y_train = []
#     num_features = training_set_scaled.shape[1]

#     for i in range(forecast_out, len(training_set_scaled)):
#         X_train.append(training_set_scaled[i-forecast_out:i, :])
#         y_train.append(target_set_scaled[i, 0])
        
#     X_train, y_train = np.array(X_train), np.array(y_train)
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))

#     # Model architecture
#     model = Sequential()
#     model.add(LSTM(unit_number, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=l2(l2_strength), activation=activation))
#     model.add(Dropout(dropout_rate))
#     model.add(BatchNormalization())
#     model.add(LSTM(unit_number, return_sequences=True, kernel_regularizer=l2(l2_strength), activation=activation))
#     model.add(Dropout(dropout_rate))
#     model.add(BatchNormalization())
#     model.add(LSTM(unit_number, activation=activation))
#     model.add(Dropout(dropout_rate))
#     model.add(BatchNormalization())
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer=optimizer)

#     # Callbacks
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=5, min_lr=min_lr)

#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

#     # Prediction on test set
#     real_stock_price = dataset_test[[target]].values
#     testing_set = sc.transform(dataset_test[features].values)
#     X_test = []
    
#     for i in range(forecast_out, len(testing_set)):
#         X_test.append(testing_set[i-forecast_out:i, :])
        
#     X_test = np.array(X_test)
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_features))
    
#     predicted_stock_price = model.predict(X_test)
#     predicted_stock_price = sc_target.inverse_transform(predicted_stock_price)
#     error_lstm = math.sqrt(mean_squared_error(real_stock_price[forecast_out:], predicted_stock_price))

#     # Recursive multi-step forecast
#     forecast_set = []
#     last_sequence = X_train[-1, :, :].reshape(1, X_train.shape[1], num_features)

#     for _ in range(forecast_out):
#         next_pred = model.predict(last_sequence)
#         forecast_set.append(float(next_pred[0, 0].item()))
        
#         next_pred_reshaped = np.zeros((1, last_sequence.shape[2]))
#         next_pred_reshaped[0, 0] = next_pred[0, 0]
        
#         new_sequence = np.concatenate((last_sequence[0, 1:, :], next_pred_reshaped), axis=0)
        
#         last_sequence = new_sequence.reshape(1, last_sequence.shape[1], last_sequence.shape[2])

#     # Inverse scaling
#     forecast_set = sc_target.inverse_transform(np.array(forecast_set).reshape(-1, 1))
    
#     # Calculate mean of forecast
#     lstm_pred = int(forecast_set[0, 0])
#     mean_forecast = int(np.mean(forecast_set))
#     lstm_forecast_set = forecast_set.flatten()

#     logger.info("+" * 50)
#     logger.info(f"D+1 LSTM Prediction for {ticker}: ${round(lstm_pred, 2)}")
#     logger.info(f"D+1 to D+{forecast_out} LSTM Mean for {ticker}: ${round(mean_forecast, 2)}")
#     logger.info(f"LSTM RMSE: {error_lstm}")
#     logger.info("=" * 50)
    
#     return lstm_pred, lstm_forecast_set, mean_forecast, error_lstm, real_stock_price[forecast_out:], predicted_stock_price

def objective_lstm(trial, df):
    forecast_out = 5
    
    # Suggest hyperparameters
    epochs = trial.suggest_int('epochs', 50, 200)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    unit_number = trial.suggest_int('unit_number', 50, 300)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    l2_strength = trial.suggest_float('l2_strength', 1e-5, 1e-2, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])
    lr_factor = trial.suggest_float('lr_factor', 0.2, 0.8, step=0.1)
    min_lr = trial.suggest_float('min_lr', 1e-5, 1e-2, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])

    # Use all columns except 'Close' for training
    features = df.columns.difference(['Close'])
    target = 'Close'
    
    # Preprocessing
    dataset_train = df.iloc[0:int(0.8*len(df)), :]
    dataset_test = df.iloc[int(0.8*len(df)):, :]
    training_set = dataset_train[features].values
    target_set = dataset_train[[target]].values
    
    sc = MinMaxScaler(feature_range=(0, 1))
    sc_target = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    target_set_scaled = sc_target.fit_transform(target_set)
    
    X_train = []
    y_train = []
    num_features = training_set_scaled.shape[1]
    
    for i in range(forecast_out, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-forecast_out:i, :])
        y_train.append(target_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))
    
    # Model building
    model = Sequential()
    model.add(LSTM(unit_number, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=l2(l2_strength), activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(unit_number, return_sequences=True, kernel_regularizer=l2(l2_strength), activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(unit_number, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=10, min_lr=min_lr)

    # Model training
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, reduce_lr])
    
    # Prediction on test set
    real_stock_price = dataset_test[[target]].values
    testing_set = sc.transform(dataset_test[features].values)
    X_test = []
    
    for i in range(forecast_out, len(testing_set)):
        X_test.append(testing_set[i-forecast_out:i, :])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_features))
    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc_target.inverse_transform(predicted_stock_price)
    
    error_lstm = math.sqrt(mean_squared_error(real_stock_price[forecast_out:], predicted_stock_price))

    return error_lstm

# def optimize_and_execute_lstm(ticker, df, n_trials=100):
#     study_lstm = optuna.create_study(direction='minimize', study_name='study_lstm')
#     study_lstm.optimize(lambda trial: objective_lstm(trial, df), n_trials=n_trials, gc_after_trial=True)
#     best_params_lstm = study_lstm.best_params
#     logger.info("+" * 50)
#     logger.info(f"Best parameters for LSTM:")
#     logger.info("=" * 50)
#     for key, value in best_params_lstm.items():
#         logger.info(f"{key}: {value}")
#     logger.info("=" * 50)
#     lstm_pred, lstm_forecast_set, mean_forecast, error_lstm = lstm_algorithm(ticker, df, best_params_lstm)
#     return lstm_pred, lstm_forecast_set, mean_forecast, error_lstm

def optimize_and_execute_lstm(ticker, df, n_trials):
    # Create a study object and specify the direction is 'minimize'.
    study = optuna.create_study(direction='minimize')

    # Progress bar setup
    st_container = st.empty()
    st_progress_bar = st_container.progress(0)

    def update_progress_bar(study, trial):
        st_progress_bar.progress((trial.number + 1) / n_trials)

    # Optimize the study, the objective function is passed in as the first argument.
    study.optimize(lambda trial: objective_lstm(trial, df), n_trials=n_trials, callbacks=[update_progress_bar], gc_after_trial=True)

    st_container.empty()
    
    st.markdown(f"<span style='color: teal'>Best Parameters from {n_trials} trial runs:</span>", unsafe_allow_html=True)
    for k, v in study.best_params.items():
        st.text(f"{k}: {v}")
    
    st.markdown("<span style='color: teal'>Now training the model with the best parameters...</span>", unsafe_allow_html=True)

    # Train and Predict using the best parameters
    lstm_pred, lstm_forecast_set, mean_forecast, error_lstm, real_stock_price, predicted_stock_price = lstm_algorithm(ticker, df, study.best_params)

    return lstm_pred, lstm_forecast_set, mean_forecast, error_lstm, real_stock_price, predicted_stock_price
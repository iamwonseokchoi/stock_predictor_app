from loguru import logger
import streamlit as st

from sklearn.metrics import mean_squared_error
import math
import numpy as np
import xgboost as xgb
import optuna
import multiprocessing
import matplotlib.pyplot as plt


def xgboost_algorithm(ticker, df, best_params):
    forecast_out = 5
    common_params = {**best_params, 'eval_metric': 'rmse'} 
    
    dataset_train = df.iloc[:int(0.8 * len(df)), :]
    dataset_test = df.iloc[int(0.8 * len(df)):, :]
    
    X_train = dataset_train.drop(['Close'], axis=1).values
    y_train = dataset_train['Close'].values
    
    X_test = dataset_test.drop(['Close'], axis=1).values
    y_test = dataset_test['Close'].values
    
    # Model Initialization
    model = xgb.XGBRegressor(**common_params)
    
    # Model Training
    model.fit(X_train, y_train)
    
    # Prediction on test set
    predicted_stock_price = model.predict(X_test)
    error_xgb = math.sqrt(mean_squared_error(y_test, predicted_stock_price))
    
    # Recursive multi-step forecast
    forecast_set = []
    last_sequence = X_test[-1]
    
    for _ in range(forecast_out):
        next_pred = model.predict(last_sequence.reshape(1, -1))
        forecast_set.append(next_pred[0])
        new_sequence = np.append(last_sequence[1:], next_pred)
        last_sequence = new_sequence
    
    # Calculate mean of forecast
    xgb_pred = float(forecast_set[0])
    mean_forecast = float(np.mean(forecast_set))
    
    # Plot
    # fig = plt.figure(figsize=(10, 5), dpi=100)
    # plt.plot(y_test, label='Actual Price')
    # plt.plot(predicted_stock_price, label='Predicted Price')
    # plt.legend(loc=4)
    # plt.show(fig)
    
    return xgb_pred, np.array(forecast_set), mean_forecast, error_xgb, y_test, predicted_stock_price

def objective_xgb(trial, df):
    
    # Hyperparameters
    params = {
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart', 'gblinear']),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'alpha': trial.suggest_float('alpha', 0.0, 1.0),
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 1.5),
        'eval_metric': 'rmse'
    }

    if params['booster'] == 'gbtree' or params['booster'] == 'dart':
        params.update({
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist'])
        })

    # Data Preparation
    dataset_train = df.iloc[:int(0.8 * len(df)), :]
    dataset_test = df.iloc[int(0.8 * len(df)):, :]
    
    X_train = dataset_train.drop(['Close'], axis=1).values
    y_train = dataset_train['Close'].values
    
    X_test = dataset_test.drop(['Close'], axis=1).values
    y_test = dataset_test['Close'].values

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    predicted_stock_price = model.predict(X_test)
    error_xgb = math.sqrt(mean_squared_error(y_test, predicted_stock_price))
    
    return error_xgb

# def optimize_and_execute_xgb(ticker, df, n_trials=100):
#     total_cores = multiprocessing.cpu_count()
#     study_xgb = optuna.create_study(direction='minimize', study_name='study_xgb')
#     study_xgb.optimize(lambda trial: objective_xgb(trial, df), n_trials=n_trials, gc_after_trial=True, n_jobs=(total_cores-2))
#     best_params_xgb = study_xgb.best_params
#     logger.info(f"Best Parameters for XGBoost: {best_params_xgb}")
#     xgb_pred, xgb_forecast_set, mean_forecast, error_xgb = xgboost_algorithm(ticker, df, best_params_xgb)
#     return xgb_pred, xgb_forecast_set, mean_forecast, error_xgb

def optimize_and_execute_xgboost(ticker, df, n_trials):
    # Create a study object and specify the direction is 'minimize'.
    study = optuna.create_study(direction='minimize')

    # Progress bar setup
    st_container = st.empty()
    st_progress_bar = st_container.progress(0)

    def update_progress_bar(study, trial):
        st_progress_bar.progress((trial.number + 1) / n_trials)

    # Optimize the study, the objective function is passed in as the first argument.
    study.optimize(lambda trial: objective_xgb(trial, df), n_trials=n_trials, callbacks=[update_progress_bar], gc_after_trial=True, n_jobs=1)

    st_container.empty()
    
    st.markdown(f"<span style='color: teal'>Best Parameters from {n_trials} trial runs:</span>", unsafe_allow_html=True)
    for k, v in study.best_params.items():
        st.text(f"{k}: {v}")
    
    st.markdown("<span style='color: teal'>Now training the model with the best parameters...</span>", unsafe_allow_html=True)

    # Train and Predict using the best parameters
    lstm_pred, lstm_forecast_set, mean_forecast, error_lstm, real_stock_price, predicted_stock_price = xgboost_algorithm(ticker, df, study.best_params)

    return lstm_pred, lstm_forecast_set, mean_forecast, error_lstm, real_stock_price, predicted_stock_price

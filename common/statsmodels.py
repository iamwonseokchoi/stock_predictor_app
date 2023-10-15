import pandas as pd
from loguru import logger

import math
import optuna
import warnings
import numpy as np
import multiprocessing
from prophet import Prophet
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


########## Linear Regression ##########
def linear_regression_algorithm(ticker, df, split_ratio, scaling_factor):
    forecast_out = int(5)
    df['Close after 5 days'] = df[['Close']].shift(-forecast_out)
    df_new = df[['Close', 'Close after 5 days']]
    y = np.array(df_new.iloc[:-forecast_out, 1])
    y = np.reshape(y, (-1, 1))
    X = np.array(df_new.iloc[:-forecast_out, :-1])
    X_forecast = np.array(df_new.iloc[-forecast_out:, :-1])
    X_train = X[:int(split_ratio*len(df)), :]
    X_test = X[int(split_ratio*len(df)):, :]
    y_train = y[:int(split_ratio*len(df)), :]
    y_test = y[int(split_ratio*len(df)):, :]
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_forecast = scaler.transform(X_forecast)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_test_pred = linear_model.predict(X_test)
    y_test_pred = y_test_pred * (scaling_factor)
    fig = plt.figure(figsize=(10, 5), dpi=100)
    # plt.title("Linear Regression Model Fit")
    # plt.plot(y_test, label='Actual Price')
    # plt.plot(y_test_pred, label='Predicted Price')
    # plt.legend(loc=4)
    # fig.savefig('./static/lin_reg.png')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#121212')  # Dark Gray
    fig.patch.set_facecolor('#121212')  # Dark Gray
    ax.plot(y_test, label='Actual Price', linewidth=2, color='blue', marker='o', markersize=5)
    ax.plot(y_test_pred, label='Predicted Price', linewidth=2, color='red', marker='x', markersize=5)
    ax.grid(True, linestyle='--', linewidth=0.5, color='white', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title('Linear Regression Model Fit', color='white', fontsize=20)
    ax.tick_params(colors='white')
    legend = ax.legend(loc=4, fontsize='large', facecolor='#121212', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")
    ax.spines[['left', 'bottom']].set_color('white')
    ax.spines[['right', 'top']].set_color('#121212')
    plt.tight_layout()
    fig.savefig('./static/lin_reg.png', bbox_inches='tight')

    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
    forecast_set = linear_model.predict(X_forecast)
    forecast_set = forecast_set * (scaling_factor)
    mean = forecast_set.mean()
    lr_pred = forecast_set[0, 0]
    forecast_set = forecast_set.flatten()

    return lr_pred, forecast_set, mean, error_lr

def objective_lr(trial, df):
    forecast_out = int(5)
    df['Close after 5 days'] = df[['Close']].shift(-forecast_out)
    split_ratio = trial.suggest_float('split_ratio', 0.7, 0.75)
    scaling_factor = trial.suggest_float('scaling_factor', 0.90, 1.10)
    df_new = df[['Close', 'Close after 5 days']]
    y = np.array(df_new.iloc[:-forecast_out, 1])
    y = np.reshape(y, (-1, 1))
    X = np.array(df_new.iloc[:-forecast_out, :-1])
    train_size = int(split_ratio * len(df))
    X_train = X[:train_size, :]
    X_test = X[train_size:, :]
    y_train = y[:train_size, :]
    y_test = y[train_size:, :]
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    linear_model = LinearRegression(n_jobs=-1)
    linear_model.fit(X_train, y_train)
    y_test_pred = linear_model.predict(X_test)
    y_test_pred = y_test_pred * scaling_factor
    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
    return error_lr

def optimize_lr(df):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_lr(trial, df), n_trials=10)
    best_params = study.best_params
    logger.info(f"Best Parameters for Linear Regression: {best_params}")
    return best_params


########## ARIMA ##########
def arima_forecast_algorithm(ticker, df, p, d, q, split_ratio, scaling_factor):
    forecast_out = 5
    df['Close after 5 days'] = df['Close'].shift(-forecast_out)
    df_new = df[['Close', 'Close after 5 days']].dropna()
    X = df_new['Close'].values
    size = int(len(X) * split_ratio)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = []
    forecast_set = []
    for t in range(len(test)):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit()
        output = model_fit.forecast(steps=forecast_out)
        yhat = output[-1]
        predictions.append(yhat * scaling_factor)
        obs = test[t]
        history.append(obs)
        history = history[1:]
    error_arima = math.sqrt(mean_squared_error(test, predictions))
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit()
    forecast_output = model_fit.forecast(steps=forecast_out)
    forecast_set = np.array(forecast_output) * scaling_factor
    arima_pred = forecast_set[0]
    mean_forecast = forecast_set.mean()
    # plt.figure(figsize=(12,6))
    # plt.plot(test, label='Actual Price')
    # plt.plot(predictions, color='red', label='Predicted')
    # plt.legend()
    # plt.show()
    
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.set_facecolor('#121212')  # Dark Gray
    # fig.patch.set_facecolor('#121212')  # Dark Gray
    # ax.plot(test, label='Actual Price', linewidth=2, color='blue', marker='o', markersize=5)
    # ax.plot(predictions, label='Predicted Price', linewidth=2, color='red', marker='x', markersize=5)
    # ax.grid(True, linestyle='--', linewidth=0.5, color='white', alpha=0.7)
    # ax.set_axisbelow(True)
    # ax.set_title('ARIMA Model Fit', color='white', fontsize=20)
    # ax.tick_params(colors='white')
    # legend = ax.legend(loc=4, fontsize='large', facecolor='#121212', edgecolor='white')
    # for text in legend.get_texts():
    #     text.set_color("white")
    # ax.spines[['left', 'bottom']].set_color('white')
    # ax.spines[['right', 'top']].set_color('#121212')
    # plt.tight_layout()
    # fig.savefig('./static/arima.png', bbox_inches='tight')

    return arima_pred, forecast_set, mean_forecast, error_arima, test, predictions

def objective_arima(trial, data):
    d = trial.suggest_int('d', 0, 2)
    max_p = 7 if d > 0 else 3
    max_q = 7 if d > 0 else 3
    p = trial.suggest_int('p', 0, max_p)
    q = trial.suggest_int('q', 0, max_q)
    split_ratio = trial.suggest_float('split_ratio', 0.7, 0.8)
    scaling_factor = trial.suggest_float('scaling_factor', 0.9, 1.1)
    train_size = int(split_ratio * len(data))
    train, test = data[:train_size], data[train_size:]
    history = [x for x in train]
    predictions = []
    try:
        for t in range(len(test)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = ARIMA(history, order=(p, d, q))
                model_fit = model.fit()
                if len(w) > 0:
                    for warn in w:
                        if "Non-invertible" in str(warn.message) or "Non-stationary" in str(warn.message):
                            return float('inf')
                output = model_fit.forecast(steps=5)
                yhat = output[-1]
                predictions.append(yhat * scaling_factor)
                obs = test[t]
                history.append(obs)
                history = history[1:]
        error_arima = math.sqrt(mean_squared_error(test, predictions))
        return error_arima
    except Exception as e:
        return float('inf')


########## Prophet ##########
def prophet_forecasting_algorithm(ticker, df, changepoint_prior_scale, seasonality_prior_scale, seasonality_mode, daily_seasonality, weekly_seasonality, yearly_seasonality):
    df_prophet = df.dropna().reset_index(drop=False)
    df_prophet = df_prophet[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    prophet_model = Prophet(
        daily_seasonality=daily_seasonality, 
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode
    )
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=5)
    forecast = prophet_model.predict(future)
    forecast['yhat'] = forecast['yhat']
    
    prophet_forecast_set = list(forecast.tail(5)['yhat'])
    prophet_pred = prophet_forecast_set[0]
    mean_forecast = sum(prophet_forecast_set)/len(prophet_forecast_set) if len(prophet_forecast_set) > 0 else 0
    error_prophet = math.sqrt(mean_squared_error(df_prophet['y'], forecast[:-5]['yhat']))
    
    # fig = plt.figure(figsize=(10, 5), dpi=100)
    # plt.plot(df_prophet['y'].values, label='Actual Price')
    # plt.plot(forecast['yhat'].values, label='Predicted Price', color='red')
    # plt.legend(loc=4)
    # plt.show(fig)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('#121212')  # Dark Gray
    fig.patch.set_facecolor('#121212')  # Dark Gray
    ax.plot(df_prophet['y'].values, label='Actual Price', linewidth=2, color='blue', marker='o', markersize=5)
    ax.plot(forecast['yhat'].values, label='Predicted Price', linewidth=2, color='red', marker='x', markersize=5)
    ax.grid(True, linestyle='--', linewidth=0.5, color='white', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title('Prophet Model Fit', color='white', fontsize=20)
    ax.tick_params(colors='white')
    legend = ax.legend(loc=4, fontsize='large', facecolor='#121212', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")
    ax.spines[['left', 'bottom']].set_color('white')
    ax.spines[['right', 'top']].set_color('#121212')
    plt.tight_layout()
    fig.savefig('./static/prophet.png', bbox_inches='tight')
    
    return prophet_pred, prophet_forecast_set, mean_forecast, error_prophet

def objective_prophet(trial, df):
    df_prophet = df.dropna().reset_index(drop=False)
    df_prophet = df_prophet[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    
    changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5)
    seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10)
    seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
    daily_seasonality = trial.suggest_categorical('daily_seasonality', [True, False])
    weekly_seasonality = trial.suggest_categorical('weekly_seasonality', [True, False])
    yearly_seasonality = trial.suggest_categorical('yearly_seasonality', [True, False])

    prophet_model = Prophet(
        daily_seasonality=daily_seasonality, 
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode
    )
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=5)
    forecast = prophet_model.predict(future)
    error_prophet = math.sqrt(mean_squared_error(df_prophet['y'], forecast[:-5]['yhat']))
    
    return error_prophet

def optimize_prophet(df):
    study = optuna.create_study(direction='minimize', study_name='study_prophet')
    study.optimize(lambda trial: objective_prophet(trial, df), n_trials=100)
    best_params = study.best_params
    logger.info(f"Best Parameters for Prophet: {best_params}")
    return best_params


########## Best Model Training ##########
# def stop_optimization_callback(study, trial):
#     RMSE_THRESHOLDS = {
#         'study_lr': 1.0,
#         'study_arima': 1.0,
#         'study_prophet': 1.0
#     }
#     if trial.value < RMSE_THRESHOLDS[study.study_name]:
#         study.stop()

def optimize_and_execute_lr(ticker, df, n_trials=100):
    total_cores = multiprocessing.cpu_count()
    study_lr = optuna.create_study(direction='minimize', study_name='study_lr')
    study_lr.optimize(lambda trial: objective_lr(trial, df), n_trials=n_trials, gc_after_trial=True, n_jobs=(total_cores-2))
    best_params_lr = study_lr.best_params
    split_ratio_best_lr = best_params_lr['split_ratio']
    scaling_factor_best_lr = best_params_lr['scaling_factor']
    return linear_regression_algorithm(ticker, df, split_ratio_best_lr, scaling_factor_best_lr)

def optimize_and_execute_arima(ticker, df, n_trials=100):
    total_cores = multiprocessing.cpu_count()
    study_arima = optuna.create_study(direction='minimize')
    study_arima.optimize(lambda trial: objective_arima(trial, df['Close'].values), 
                            n_trials=n_trials, gc_after_trial=True, n_jobs=(total_cores - 2))
    best_params_arima = study_arima.best_params
    return arima_forecast_algorithm(
        ticker, df, best_params_arima['p'], best_params_arima['d'], best_params_arima['q'], 
        best_params_arima['split_ratio'], best_params_arima['scaling_factor'])

def optimize_and_execute_prophet(ticker, df, n_trials=20):
    total_cores = multiprocessing.cpu_count()
    study_prophet = optuna.create_study(direction='minimize', study_name='study_prophet')
    study_prophet.optimize(lambda trial: objective_prophet(trial, df), n_trials=n_trials, gc_after_trial=True, n_jobs=(total_cores-2))
    best_params_prophet = study_prophet.best_params
    changepoint_best = best_params_prophet['changepoint_prior_scale']
    seasonality_prior_best = best_params_prophet['seasonality_prior_scale']
    seasonality_mode_best = best_params_prophet['seasonality_mode']
    daily_seasonality_best = best_params_prophet['daily_seasonality']
    weekly_seasonality_best = best_params_prophet['weekly_seasonality']
    yearly_seasonality_best = best_params_prophet['yearly_seasonality']
    return prophet_forecasting_algorithm(
        ticker, 
        df, 
        changepoint_best, 
        seasonality_prior_best, 
        seasonality_mode_best,
        daily_seasonality_best,
        weekly_seasonality_best,
        yearly_seasonality_best
    )
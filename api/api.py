from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from common.functions import *
from common.statsmodels import *
from common.nnmodels import *
from common.xgbmodel import *


app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/logs", response_class=PlainTextResponse)
async def get_logs():
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            logs = f.read()
        return logs
    else:
        return "Log file does not exist."


@app.get("/all_tickers")
async def get_all_tickers():
    df = pd.read_csv("data/nasdaq_listings.csv", usecols=['Symbol', 'Company Name'])
    ticker_name_json = df.to_json(orient="records")
    return JSONResponse(content=ticker_name_json)


@app.get("/info/{ticker}")
async def get_company_info(ticker):
    return JSONResponse(fetch_company_info(ticker))


@app.get("/ohlcv/{ticker}")
async def get_ohlcv(ticker):
    df = fetch_ohlcv(ticker)
    return JSONResponse(content=df.to_json(orient="split"))


@app.get("/ml-dataset/{ticker}")
async def get_ml_dataset(ticker: str):
    df = fetch_ml_dataframe(ticker)
    return JSONResponse(content=df.to_json(orient="split"))


@app.get("/train/statsmodels/lr/{ticker}")
async def run_linear_regression(ticker: str):
    df = fetch_ml_dataframe(ticker)
    _, lr_pred, lr_forecast_set, mean_forecast_lr, error_lr = optimize_and_execute_lr(ticker, df, n_trials=100)
    forecast_df = create_forecast_dataframe(df, forecast_set=lr_forecast_set, type='Lin_Reg')
    payload = {
        'forecast_df': forecast_df.to_json(orient="split"),
        'next_day_forecast': lr_pred,
        '5_day_mean_forecast': mean_forecast_lr,
        'msre': error_lr
    }
    return JSONResponse(content=payload)


@app.get("/train/statsmodels/arima/{ticker}")
async def run_arima(ticker: str):
    df = fetch_ml_dataframe(ticker)
    _, arima_pred, arima_forecast_set, mean_forecast_arima, error_arima = optimize_and_execute_arima(ticker, df, n_trials=5)
    forecast_df = create_forecast_dataframe(df, forecast_set=arima_forecast_set, type='ARIMA')
    payload = {
        'forecast_df': forecast_df.to_json(orient="split"),
        'next_day_forecast': arima_pred,
        '5_day_mean_forecast': mean_forecast_arima,
        'msre': error_arima
    }
    return JSONResponse(content=payload)


@app.get("/train/statsmodels/prophet/{ticker}")
async def run_prophet(ticker: str):
    df = fetch_ml_dataframe(ticker)
    _, prophet_pred, prophet_forecast_set, mean_forecast_prophet, error_prophet = optimize_and_execute_prophet(ticker, df, n_trials=10)
    forecast_df = create_forecast_dataframe(df, forecast_set=prophet_forecast_set, type='Prophet')
    payload = {
        'forecast_df': forecast_df.to_json(orient="split"),
        'next_day_forecast': prophet_pred,
        '5_day_mean_forecast': mean_forecast_prophet,
        'msre': error_prophet
    }
    return JSONResponse(content=payload)


@app.get("/train/nnmodels/lstm/{ticker}")
async def run_lstm(ticker: str):
    df = fetch_ml_dataframe(ticker)
    lstm_pred, lstm_forecast_set, mean_forecast_lstm, error_lstm = optimize_and_execute_lstm(ticker, df, n_trials=5)
    forecast_df = create_forecast_dataframe(df, forecast_set=lstm_forecast_set, type='LSTM')
    payload = {
        'forecast_df': forecast_df.to_json(orient="split"),
        'next_day_forecast': lstm_pred,
        '5_day_mean_forecast': mean_forecast_lstm,
        'msre': error_lstm
    }
    return JSONResponse(content=payload)


@app.get("/train/xgbmodel/xgb/{ticker}")
async def run_xgb(ticker: str):
    df = fetch_ml_dataframe(ticker)
    xgb_pred, xgb_forecast_set, mean_forecast_xgb, error_xgb = optimize_and_execute_xgb(ticker, df, n_trials=100)
    forecast_df = create_forecast_dataframe(df, forecast_set=xgb_forecast_set, type='XGBoost')
    payload = {
        'forecast_df': forecast_df.to_json(orient="split"),
        'next_day_forecast': xgb_pred,
        '5_day_mean_forecast': mean_forecast_xgb,
        'msre': error_xgb
    }
    return JSONResponse(content=payload)
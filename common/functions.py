import os
import sys
import requests
import warnings
import pandas as pd
import datetime as dt
import yfinance as yf
from loguru import logger
from decouple import config

import numpy as np
import pandas_market_calendars as mcal
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import CCIIndicator, ADXIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator


########## LOGGING ##########
log_path = config('LOG_PATH', default='./logs/sys_logs.log')
log_dir = os.path.dirname(log_path)

if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(log_path):
    with open(log_path, 'w'): pass

logger.add(log_path, rotation="500 MB", retention="10 days", compression="zip")
logger.add(sys.stderr, colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | \
        <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


########## DATA ENGINEERING ##########
def fetch_fred_data():
    api_key = config('FRED_API_KEY')
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    SERIES_IDS = [
        'DTB3',      # 3-Month Treasury Bill
        'GS10',      # 10-Year Treasury Constant Maturity Rate
        'UNRATE',    # Unemployment Rate
        'CPIAUCSL',  # Consumer Price Index for All Urban Consumers
        'GDP',       # Gross Domestic Product
        'SP500',     # S&P 500
        'FEDFUNDS'   # Effective Federal Funds Rate
    ]
    aggregated_df = pd.DataFrame()
    for series_id in SERIES_IDS:
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json'
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()["observations"]
        temp_df = pd.DataFrame(data)
        temp_df = temp_df.set_index('date')
        temp_df = temp_df[['value']].rename(columns={'value': series_id})
        if aggregated_df.empty:
            aggregated_df = temp_df
        else:
            aggregated_df = aggregated_df.merge(temp_df, left_index=True, right_index=True, how='outer')
    aggregated_df = aggregated_df.ffill().dropna()
    return aggregated_df

def fetch_company_info(ticker):
    stock_data = yf.Ticker(ticker)
    company_info = stock_data.info
    return company_info

def fetch_ohlcv(ticker):
    stock_data = yf.Ticker(ticker)
    ohlcv = stock_data.history(period="5y")
    ohlcv.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    return ohlcv

def apply_technicals(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        williams_r = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close']).williams_r()
        stochastic_oscillator = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        roc = ROCIndicator(df['Close']).roc()
        cci = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close']).cci()
        adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
        ema20 = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        ema50 = EMAIndicator(close=df['Close'], window=50).ema_indicator()
        ema100 = EMAIndicator(close=df['Close'], window=100).ema_indicator()
        bollinger = BollingerBands(close=df['Close'])
        vwap = np.cumsum(df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3) / np.cumsum(df['Volume'])
        rsi = RSIIndicator(close=df['Close']).rsi()
        macd = MACD(close=df['Close'])
        dmp_16 = adx_indicator.adx_pos()
        df['williams_r'] = williams_r
        df['stoch_k'] = stochastic_oscillator.stoch()
        df['roc'] = roc
        df['cci'] = cci
        df['adx'] = adx_indicator.adx()
        df['atr'] = atr
        df['20D-EMA'] = ema20
        df['50D-EMA'] = ema50
        df['100D-EMA'] = ema100
        df['bollinger_mavg'] = bollinger.bollinger_mavg()
        df['bollinger_hband'] = bollinger.bollinger_hband()
        df['bollinger_lband'] = bollinger.bollinger_lband()
        df['vwap'] = vwap
        df['pvwap'] = df['Close'] - vwap
        df['rsi'] = rsi
        df['rsicat'] = pd.cut(rsi, bins=[0, 30, 70, 100], labels=[1, 2, 3], right=False).astype('double')
        df.reset_index(inplace=True)
        df['dayofweek'] = df['Date'].dt.dayofweek + 1
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df.set_index('Date', inplace=True)
        df['MACDs_12_26_9'] = macd.macd_signal()
        df['MACDh_12_26_9'] = macd.macd_diff()
        df['DMP_16'] = dmp_16
        df = df.bfill()
    return df

def fetch_master_dataframe(ticker):
    stock_data = yf.Ticker(ticker)
    df = stock_data.history(period="max")
    if "Dividends" in df.columns and "Stock Splits" in df.columns:
        df.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    nasdaq = yf.Ticker("^IXIC").history(period="max").drop(columns=["Dividends", "Stock Splits"])
    nasdaq.columns = [f"{col}_nasdaq" for col in nasdaq.columns]
    df = df.join(nasdaq, how="left")
    return df.sort_index()

def fetch_combined_dataframe(ticker):
    master_df = fetch_master_dataframe(ticker)
    fred_df = fetch_fred_data()
    fred_df.replace('.', pd.NA, inplace=True)
    fred_df.ffill(inplace=True)
    if fred_df.isnull().values.any():
        fred_df.bfill(inplace=True)
    master_df.index = pd.to_datetime(master_df.index).normalize().tz_localize(None)
    fred_df.index = pd.to_datetime(fred_df.index)
    combined_df = master_df.join(fred_df, how='left')
    columns_to_convert = ['DTB3', 'GS10', 'UNRATE', 'CPIAUCSL', 'GDP', 'SP500', 'FEDFUNDS']
    for column in columns_to_convert:
        combined_df[column] = combined_df[column].astype(float)
    output_df = apply_technicals(combined_df)
    return output_df

def fetch_ml_dataframe(ticker):
    ml_range = (dt.datetime.today() - dt.timedelta(days=365*5)).strftime("%Y-%m-%d")
    df = fetch_combined_dataframe(ticker).query(f"Date >= '{ml_range}'")
    if df.isnull().values.any():
        df = df.dropna()
    return df


########## Misc ##########
def replace_negatives_with_last_positive(col):
    last_positive = None
    for idx, value in enumerate(col):
        if value >= 0:
            last_positive = value
        elif last_positive is not None:
            col[idx] = last_positive
    return col

def create_forecast_dataframe(df, forecast_set, type: str):
    last_date = df.index.max()
    last_close_value = df.loc[last_date, 'Close']
    nyse_cal = mcal.get_calendar('NYSE')
    next_trading_days = nyse_cal.valid_days(
        start_date=last_date + pd.Timedelta(days=1), end_date=last_date + pd.Timedelta(days=30))
    next_5_trading_days = next_trading_days[:5].tz_localize(None)
    combined_dates = pd.to_datetime([last_date]).append(next_5_trading_days)
    combined_forecast = [last_close_value] + list(forecast_set)
    
    capitalized_type = type.capitalize()
    
    forecast_df = pd.DataFrame({
        'Date': combined_dates,
        f'{capitalized_type}_Forecast': combined_forecast
    })
    forecast_df.set_index('Date', inplace=True)
    forecast_df = forecast_df.apply(replace_negatives_with_last_positive)
    return forecast_df
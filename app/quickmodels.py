import streamlit as st
import pandas as pd

from common.statsmodels import *
from common.functions import *

@st.cache_data
def cache_ml_dataframe(search_ticker):
    with st.spinner('Loading Data...'):
        df_ml = fetch_ohlcv(search_ticker)
    return df_ml

def get_ticker_and_data(search_term):
    listings = pd.read_csv("data/nasdaq_listings.csv")["Symbol"].to_list()
    search_ticker = search_term.upper()
    df_ml = None

    if search_ticker in listings:
        df_ml = cache_ml_dataframe(search_ticker)
    elif search_ticker not in listings and search_ticker:
        st.warning(f"{search_ticker} not in NASDAQ. Please enter a listed NASDAQ ticker symbol to continue.")
        
    st.cache_data.clear()
    return search_ticker, df_ml

def get_search():
    search_term = st.text_input("Search by Ticker Symbol to perform Linear Regression & Prophet prediction", "")
    search_ticker, df_ml = get_ticker_and_data(search_term)
    return search_ticker, df_ml


def predict_statsmodels():
    search_ticker, df_ml = get_search()
    
    if df_ml is not None:
        open_last = df_ml['Open'].iloc[-1]
        open_second_last = df_ml['Open'].iloc[-2]
        close_last = df_ml['Close'].iloc[-1]
        close_second_last = df_ml['Close'].iloc[-2]
        high_last = df_ml['High'].iloc[-1]
        high_second_last = df_ml['High'].iloc[-2]
        low_last = df_ml['Low'].iloc[-1]
        low_second_last = df_ml['Low'].iloc[-2]
        volume_last = df_ml['Volume'].iloc[-1]
        volume_second_last = df_ml['Volume'].iloc[-2]

        # Calculate the percentage differences
        delta_open = ((open_last - open_second_last) / open_second_last) * 100
        delta_close = ((close_last - close_second_last) / close_second_last) * 100
        delta_high = ((high_last - high_second_last) / high_second_last) * 100
        delta_low = ((low_last - low_second_last) / low_second_last) * 100
        delta_volume = ((volume_last - volume_second_last) / volume_second_last) * 100

        st.markdown(f'<h3 style="color:teal;">{search_ticker} Today</h3>', unsafe_allow_html=True)
        o, c, h, l, v = st.columns(5)

        # Display the metrics with the calculated deltas
        with o:
            st.metric("Open", f"${open_last:,.2f}", delta=f"{delta_open:.2f}%")
        with c:
            st.metric("Close", f"${close_last:,.2f}", delta=f"{delta_close:.2f}%")
        with h:
            st.metric("High", f"${high_last:,.2f}", delta=f"{delta_high:.2f}%")
        with l:
            st.metric("Low", f"${low_last:,.2f}", delta=f"{delta_low:.2f}%")
        with v:
            st.metric("Volume", f"{volume_last:,}", delta=f"{delta_volume:.2f}%")

        st.markdown("---")
        
        c1, c2 = st.columns(2)
        # Lin Reg
        with c1:
            st.subheader("Linear Regression Model")
            split_ratio_lr = 0.8
            scaling_factor_lr = 1.00
            split_ratio_lr = st.slider("Train/Test Split Ratio", 0.5, 1.0, 0.8, 0.01)
            scaling_factor_lr = st.slider("Apply Scaling Multiplier", 0.8, 1.2, 1.00, 0.01)
            try: 
                lr_pred, lr_forecast, lr_mean, lr_error = linear_regression_algorithm(
                    search_ticker, df_ml, split_ratio_lr, scaling_factor_lr)
            except Exception:
                lr_pred, lr_forecast, lr_mean, lr_error = None, None, None, None
            st.image("static/lin_reg.png")
        # Prophet
        with c2:
            st.subheader("Prophet Model")
            changepoint_prior_scale = 0.05
            seasonality_prior_scale = 10
            seasonality_mode = "multiplicative"
            daily_seasonality = False
            weekly_seasonality = False
            yearly_seasonality = True
            changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.01, 0.85, 0.05, 0.01)
            seasonality_prior_scale = st.slider("Seasonality Prior Scale", 0.01, 50.0, 10.0, 0.01)
            try: 
                prophet_pred, prophet_forecast, prophet_mean, prophet_error = prophet_forecasting_algorithm(
                    search_ticker, df_ml, changepoint_prior_scale, seasonality_prior_scale, seasonality_mode, 
                    daily_seasonality, weekly_seasonality, yearly_seasonality)
            except Exception:
                prophet_pred, prophet_forecast, prophet_mean, prophet_error = None, None, None, None
            st.image("static/prophet.png")
        
        # Next Day Forecast
        f1, f2 = st.columns(2)
        with f1:
            t1, t2 = st.columns(2)
            with t1:
                st.markdown(f'<div style="text-align: center;color: cyan;">Linear Regression Forecast (Next Close)<br><span style="font-size: 2em;">${(float(lr_pred)):,.2f}</span></div>', unsafe_allow_html=True)
            with t2:
                st.markdown(f'<div style="text-align: center;color: orange;">Linear Regression Model RMSE<br><span style="font-size: 2em;">{lr_error:.2f}</span></div>', unsafe_allow_html=True)
        with f2:
            t1, t2 = st.columns(2)
            with t1:
                st.markdown(f'<div style="text-align: center;color: cyan;">Prophet Forecast (Next Close)<br><span style="font-size: 2em;">${prophet_pred:,.2f}</span></div>', unsafe_allow_html=True)
            with t2:
                st.markdown(f'<div style="text-align: center;color: orange;">Prophet Model RMSE<br><span style="font-size: 2em;">{prophet_error:.2f}</span></div>', unsafe_allow_html=True)
        
    return 
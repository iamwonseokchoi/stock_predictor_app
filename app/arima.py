import streamlit as st
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from pmdarima import auto_arima

from common.functions import *
from common.statsmodels import *


def valid_search_term(search_term, listings):
    return search_term.upper() in listings['Symbol'].str.upper().tolist()

def native_acf_pacf_plot(df):
    lag_acf = acf(df['Close'], nlags=30)
    lag_pacf = pacf(df['Close'], nlags=30, method='ols')

    acf_data = pd.DataFrame({'Lags': range(31), 'ACF': lag_acf})
    pacf_data = pd.DataFrame({'Lags': range(31), 'PACF': lag_pacf})

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(acf_data.set_index('Lags'))

    with col2:
        st.bar_chart(pacf_data.set_index('Lags'))

def predict_arima():
    nasdaq_listings = pd.read_csv("data/nasdaq_listings.csv")
    
    search_term = st.text_input("Search by Ticker Symbol to perform ARIMA prediction", "")
    if search_term:
        if not valid_search_term(search_term, nasdaq_listings):
            st.warning("Please input a valid listed NASDAQ ticker symbol.")
            return

        df = fetch_ml_dataframe(search_term)
        
        if df is not None:
            open_last = df['Open'].iloc[-1]
            open_second_last = df['Open'].iloc[-2]
            close_last = df['Close'].iloc[-1]
            close_second_last = df['Close'].iloc[-2]
            high_last = df['High'].iloc[-1]
            high_second_last = df['High'].iloc[-2]
            low_last = df['Low'].iloc[-1]
            low_second_last = df['Low'].iloc[-2]
            volume_last = df['Volume'].iloc[-1]
            volume_second_last = df['Volume'].iloc[-2]

            # Calculate the percentage differences
            delta_open = ((open_last - open_second_last) / open_second_last) * 100
            delta_close = ((close_last - close_second_last) / close_second_last) * 100
            delta_high = ((high_last - high_second_last) / high_second_last) * 100
            delta_low = ((low_last - low_second_last) / low_second_last) * 100
            delta_volume = ((volume_last - volume_second_last) / volume_second_last) * 100

            st.markdown(f'<h3 style="color:teal;">{search_term} Today</h3>', unsafe_allow_html=True)
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
        
        st.markdown(f'<h3 style="color:teal; font-size: 24px;">Mini Exploratory Analysis for {search_term}</h3>', unsafe_allow_html=True)
        st.line_chart(df['Close'])
        
        st.markdown(f'<h3 style="color:teal; font-size: 18px;">ACF & PACF for {search_term}</h3>', unsafe_allow_html=True)
        native_acf_pacf_plot(df)
        
        # Optimal p, d, q values using auto_arima
        model = auto_arima(df['Close'], seasonal=False, trace=True)
        st.markdown(f'<h3 style="color:cyan; font-size: 24px;">Optimal {search_term} (p, d, q): {model.order}</h3>', unsafe_allow_html=True)
        
        p = model.order[0]
        d = model.order[1]
        q = model.order[2]

        split_ratio = st.slider("Select Train/Test Split Ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.01)
        scaling_factor = st.slider("Select Scaling Multiplier", min_value=0.5, max_value=1.5, value=1.05, step=0.01)
        
        if st.button("Run ARIMA"):
            with st.spinner(f"Running ARIMA model... Using p={p}, d={d}, q={q}, Train split: {split_ratio}, Scale factor: {scaling_factor}"):
                arima_pred, forecast_set, mean_forecast, error_arima, test, predictions = arima_forecast_algorithm(
                    search_term, df, p, d, q, split_ratio, scaling_factor)
            
            st.line_chart(pd.DataFrame({'Actual': test, 'Predicted': predictions}))
            
            st.markdown(f"<div style='text-align: center; border-radius: 25px; background: #00aa00; padding: 10px;'>\
                            <h1 style='color: white;'>ARIMA Prediction for {search_term} (Next Close): ${round(arima_pred, 2)}</h1>\
                            <h1 style='color: white;'>Error (RMSE): {round(error_arima, 2)}</h1>\
                        </div>", unsafe_allow_html=True)

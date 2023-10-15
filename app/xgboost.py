import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import math

from common.functions import *
from common.xgbmodel import *
from app.arima import valid_search_term

def predict_xgboost():
    nasdaq_listings = pd.read_csv("data/nasdaq_listings.csv")
    
    search_term = st.text_input("Search by Ticker Symbol to perform XGBoost prediction", "")
    
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
        
        st.markdown(f'<h3 style="color:teal; font-size: 24px;">{search_term} Close Prices (5-yr)</h3>', unsafe_allow_html=True)
        st.line_chart(df['Close'])

        # XGBoost Parameters
        n_estimators = st.slider("Select Number of Estimators", 50, 500, 100)
        learning_rate = st.slider("Select Learning Rate", 0.01, 1.0, 0.50)
        max_depth = st.slider("Select Max Depth", 2, 20, 5)
        booster = st.selectbox("Select Booster", ["gblinear", "gbtree", "dart"])

        best_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'booster': booster,
        }

        if st.button("Run XGBoost"):
            with st.spinner(f"Training XGBoost model..."):
                xgb_pred, xgb_forecast_set, xgb_mean, error_xgb, y_test, predicted_stock_price = xgboost_algorithm(search_term, df, best_params)
                
            real_stock_price = df['Close'].iloc[int(0.8 * len(df)):].values
            
            st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': predicted_stock_price}))
            
            st.markdown(f"<div style='text-align: center; border-radius: 25px; background: #00aa00; padding: 10px;'>\
                            <h1 style='color: white;'>XGBoost Prediction (Next Close): ${xgb_pred:.2f}</h1>\
                            <h1 style='color: white;'>Error (RMSE): {error_xgb:.2f}</h1>\
                        </div>", unsafe_allow_html=True)


def auto_xgb():
    nasdaq_listings = pd.read_csv("data/nasdaq_listings.csv")
    search_term = st.text_input("Enter NASDAQ Ticker for Auto XGBoost", "")
        
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
    
        df = fetch_ml_dataframe(search_term)
        n_trials = st.number_input('Enter number of Optuna trials', min_value=1, max_value=5, value=5)
        if st.button("Run Auto XGBoost"):
            with st.spinner(f"Training XGBoost model..."):
                xgb_pred, xgb_forecast_set, xgb_mean, error_xgb, y_test, predicted_stock_price = optimize_and_execute_xgboost(ticker=search_term, df=df, n_trials=n_trials)
                
            real_stock_price = df['Close'].iloc[int(0.8 * len(df)):].values
            
            st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': predicted_stock_price}))
            
            st.markdown(f"<div style='text-align: center; border-radius: 25px; background: #00aa00; padding: 10px;'>\
                            <h1 style='color: white;'>XGBoost Prediction (Next Close): ${xgb_pred:.2f}</h1>\
                            <h1 style='color: white;'>Error (RMSE): {error_xgb:.2f}</h1>\
                        </div>", unsafe_allow_html=True)

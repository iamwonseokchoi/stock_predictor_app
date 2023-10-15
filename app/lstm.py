import streamlit as st
import pandas as pd
import numpy as np

from common.functions import *
from app.arima import valid_search_term
from common.nnmodels import *


def predict_lstm():
    nasdaq_listings = pd.read_csv("data/nasdaq_listings.csv")
    
    search_term = st.text_input("Search by Ticker Symbol to perform LSTM prediction", "")
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

        # LSTM Parameters for Optuna Study
        epochs = st.slider("Select Number of Epochs", 5, 50, 10)
        batch_size = st.slider("Select Batch Size", 20, 100, 32)
        unit_number = st.slider("Select Number of Units", 50, 200, 100)
        dropout_rate = st.slider("Select Dropout Rate", 0.1, 0.5, 0.2)
        lr_factor = st.slider("Select Learning Rate Reduction Factor", 0.1, 0.9, 0.5)
        activation = st.selectbox("Select Activation Function", ['tanh', 'relu', 'sigmoid'])
        optimizer = st.selectbox("Select Optimizer", ["Adam", "SGD", "RMSprop"])

        best_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'unit_number': unit_number,
            'dropout_rate': dropout_rate,
            'l2_strength': 0.0001,
            'activation': activation,
            'lr_factor': lr_factor,
            'min_lr': 0.0001,
            'optimizer': optimizer
        }

        if st.button("Run LSTM"):
            with st.spinner(f"Training LSTM model for {epochs} epochs... Note: Training will exit early if validation loss does not improve for 15 epochs."):
                # Call lstm_algorithm function
                lstm_pred, lstm_forecast_set, lstm_mean, error_lstm, real_stock_price, predicted_stock_price = lstm_algorithm(search_term, df, best_params)
            
            real_stock_price = real_stock_price.flatten()
            predicted_stock_price = predicted_stock_price.flatten()
            
            st.line_chart(pd.DataFrame({'Actual': real_stock_price, 'Predicted': predicted_stock_price}))
            
            st.markdown(f"<div style='text-align: center; border-radius: 25px; background: #00aa00; padding: 10px;'>\
                            <h1 style='color: white;'>LSTM Prediction (Next Close): ${lstm_pred:.2f}</h1>\
                            <h1 style='color: white;'>Error (RMSE): {error_lstm:.2f}</h1>\
                        </div>", unsafe_allow_html=True)


def auto_lstm():
    nasdaq_listings = pd.read_csv("data/nasdaq_listings.csv")
    search_term = st.text_input("Enter NASDAQ Ticker for Auto LSTM", "")
        
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
        n_trials = st.number_input('Enter number of Optuna trials', min_value=1, max_value=10, value=5)
        if st.button('Run Auto LSTM'):
            with st.spinner(f"Finding best hyperparameters within {n_trials} Trials..."):
                lstm_pred, lstm_forecast_set, mean_forecast, error_lstm, real_stock_price, predicted_stock_price = optimize_and_execute_lstm(ticker=search_term, df=df, n_trials=n_trials)
                # Display results to the user
            real_stock_price = real_stock_price.flatten()
            predicted_stock_price = predicted_stock_price.flatten()
            
            st.line_chart(pd.DataFrame({'Actual': real_stock_price, 'Predicted': predicted_stock_price}))
            
            st.markdown(f"<div style='text-align: center; border-radius: 25px; background: #00aa00; padding: 10px;'>\
                            <h1 style='color: white;'>LSTM Prediction (Next Close): ${lstm_pred:.2f}</h1>\
                            <h1 style='color: white;'>Error (RMSE): {error_lstm:.2f}</h1>\
                        </div>", unsafe_allow_html=True)

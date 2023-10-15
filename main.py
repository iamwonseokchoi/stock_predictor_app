from common.nnmodels import *
from common.xgbmodel import *
from common.functions import *
from common.statsmodels import *
from app.mainpage import *
from app.quickmodels import *
from app.arima import *
from app.lstm import *
from app.xgboost import *

import streamlit as st


# Set page width
st.set_page_config(layout="wide")
# Main tabs
Main, StatsModels, Arima, Lstm, Xgboost, OptunaLstm, OptunaXgboost = st.tabs(["Home", "Linear Regression/Prophet", "ARIMA", "LSTM", "XGBoost", "Auto LSTM", "Auto XGBoost"])

# ---------- Functions ----------
current_time, is_open = is_market_open()
if is_open:
    open_close = "Open"
else:
    open_close = "Closed"

# ASCII "button" using Unicode circle, and setting font-size to make it larger
ascii_button_color = "green" if is_open else "red"
ascii_button = f"<span style='color: {ascii_button_color}; font-size: 18px;'>●</span>"

# Intro Text
centered_text = '''
    <div style="text-align: center; font-size: 18px;">
        Mini-app that offers various statistical and machine learning tools to forecast NASDAQ listed stocks, and also train your own models. Feel free to explore!<br>
        <span style="font-style: bold;">Author: Wonseok Choi (<a href="https://www.linkedin.com/in/wonseok-c-387b57226/" target="_blank">Linkedin</a>)</span>
    </div>
'''

# ---------- Main Tab ----------
with Main: 
    _, header_col2 = st.columns([1, 1])

    st.markdown("<h1 style='text-align: center; color: #FFA500;'>Generating NASDAQ Alpha</h1>", unsafe_allow_html=True)
    with header_col2: 
        st.markdown(f"<div style='text-align: right;'>{current_time} EST / Market {open_close} {ascii_button}</div>", unsafe_allow_html=True)
    st.markdown(centered_text, unsafe_allow_html=True)
    
    # Search company for dashboard view
    search_company()


# ---------- Quick Predictions Tab ----------
with StatsModels:
    st.markdown(f"<div style='text-align: right;'>{current_time} EST / Market {open_close} {ascii_button}</div>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #FFA500; font-size: 30px;'>Linear Regression & Prophet Forecasting</h1>", unsafe_allow_html=True)
    # Initiate target for prediction
    predict_statsmodels()


# ---------- ARIMA Tab ----------
with Arima:
    st.markdown(f"<div style='text-align: right;'>{current_time} EST / Market {open_close} {ascii_button}</div>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #FFA500; font-size: 30px;'>ARIMA Forecasting</h1>", unsafe_allow_html=True)
    # Initiate target for prediction
    predict_arima()


# ---------- LSTM Tab ----------
with Lstm:
    st.markdown(f"<div style='text-align: right;'>{current_time} EST / Market {open_close} {ascii_button}</div>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #FFA500; font-size: 30px;'>LSTM Forecasting</h1>", unsafe_allow_html=True)
    # Initiate target for prediction
    predict_lstm()


# ---------- XGBoost Tab ----------
with Xgboost:
    st.markdown(f"<div style='text-align: right;'>{current_time} EST / Market {open_close} {ascii_button}</div>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #FFA500; font-size: 30px;'>XGBoost Forecasting</h1>", unsafe_allow_html=True)
    # Initiate target for prediction
    predict_xgboost()


# ---------- Optuna LSTM Tab ----------
with OptunaLstm:
    st.markdown(f"<div style='text-align: right;'>{current_time} EST / Market {open_close} {ascii_button}</div>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #FFA500; font-size: 30px;'>Auto LSTM</h1>", unsafe_allow_html=True)
    # Initiate target for optuna
    auto_lstm()
    

# ---------- Optuna XGBoost Tab ----------
with OptunaXgboost:
    st.markdown(f"<div style='text-align: right;'>{current_time} EST / Market {open_close} {ascii_button}</div>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #FFA500; font-size: 30px;'>Auto XGBoost</h1>", unsafe_allow_html=True)
    # Initiate target for optuna
    auto_xgb()

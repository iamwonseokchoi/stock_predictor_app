import pandas_market_calendars as mcal
from datetime import datetime
from pytz import timezone
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from common.functions import *
from common.statsmodels import *
from common.nnmodels import *
from common.xgbmodel import *

def is_market_open():
    # Get the NASDAQ calendar
    nasdaq = mcal.get_calendar('XNYS')
    
    # Get today's date in US/Eastern timezone
    eastern = timezone('US/Eastern')
    current_time = datetime.now(eastern)
    
    # Get market open and close times for today
    market_schedule = nasdaq.schedule(start_date=current_time, end_date=current_time)
    
    if market_schedule.empty:
        return current_time, False

    market_open = market_schedule.iloc[0]['market_open'].astimezone(eastern)
    market_close = market_schedule.iloc[0]['market_close'].astimezone(eastern)
    
    # Check if the current time is within market hours
    is_open = market_open <= current_time <= market_close
    human_time = current_time.strftime("%Y-%m-%d %H:%M")
    return human_time, is_open


def search_company():
    df = pd.read_csv("data/nasdaq_listings.csv")
    search_term = st.text_input("Search by Ticker Symbol or Company Name", "")
    if search_term:
        filtered_df = df[
            df.apply(lambda row: search_term.lower() in ' '.join(row.astype(str).str.lower()).split(), axis=1)]
        if not filtered_df.empty:
            selected_symbol = filtered_df.iloc[0]['Symbol']
            st.write(f"Displaying: {selected_symbol}")

            # Fetch OHLCV data
            stock_data = yf.download(selected_symbol, period="6mo")

            # Create trend line
            z = np.polyfit(range(len(stock_data['Close'])), stock_data['Close'], 1)
            p = np.poly1d(z)
            trendline = p(range(len(stock_data['Close'])))

            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                open=stock_data['Open'],
                                                high=stock_data['High'],
                                                low=stock_data['Low'],
                                                close=stock_data['Close']),
                                    go.Scatter(x=stock_data.index, y=trendline, mode='lines', name='Trendline')])

            fig.update_layout(title=f"{selected_symbol} 6-mo",
                                xaxis_rangeslider_visible=False,
                                showlegend=False)

            st.plotly_chart(fig, use_container_width=True)

            # Fetch and display stock info
            stock_info = yf.Ticker(selected_symbol).info
            daily_gain_loss = round(((stock_info.get('previousClose', 0) - stock_data['Open'].iloc[-1]) / stock_data['Open'].iloc[-1]) * 100, 2)
            trail_pe = round(stock_info.get('trailingPE', 'N/A'), 2) if isinstance(stock_info.get('trailingPE', None), (int, float)) else 'N/A'
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Prev. Close", f"${stock_info.get('previousClose', 'N/A')}", delta=daily_gain_loss)
            with col2:
                st.metric("Forward P/E", f"{stock_info.get('forwardPE', 'N/A')}", delta=(trail_pe))
            with col3:
                st.metric("Market Cap", f"$ {stock_info.get('marketCap', 0) / 1_000_000_000:,.0f} B")
            with col4:
                st.metric("Beta", stock_info.get('beta', 'N/A'))

            # Display company info
            st.markdown("---")
            st.markdown(f"### **{stock_info['longName']}**")
            st.markdown(f"Industry: {stock_info['industry']} / Website: {stock_info['website']}")
            st.markdown(f"Description: {stock_info['longBusinessSummary']}")

        else:
            st.warning("No results found. Please make sure the search is listed in NASDAQ.")

    else:
        # Default to NASDAQ index (symbol "^IXIC") if no search term is provided
        stock_data = yf.download("^IXIC", period="6mo")
        fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                            open=stock_data['Open'],
                                            high=stock_data['High'],
                                            low=stock_data['Low'],
                                            close=stock_data['Close'])])
        fig.update_layout(title="NASDAQ Index 6-mo",
                            xaxis_rangeslider_visible=False,
                            showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

        # Default metrics for NASDAQ
        stock_info = yf.Ticker("^IXIC").info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("52-Week Low", f"$ {stock_info.get('fiftyTwoWeekLow', 0):,}")
        with col2:
            st.metric("52-Week High", f"$ {stock_info.get('fiftyTwoWeekHigh', 0):,}")
        with col3:
            st.metric("50-Day Average", f"$ {stock_info.get('fiftyDayAverage', 0):,}")
        with col4:
            st.metric("Prev. Close", f"$ {stock_info.get('previousClose', 0):,}")
    
    return

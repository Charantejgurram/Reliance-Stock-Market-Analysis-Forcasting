import streamlit as st
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import math

import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor






st.set_page_config(layout="wide",page_title="Reliance Stock Price Prediction App")

hide_default_format = """
       <style>
       .css-z5fcl4 {padding: 0px 90px;}
       footer {visibility: hidden;}
       .css-18ni7ap {margin-top: -50px}
       .css-10trblm {margin-top: 8px}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

############## Data Fetching #####################
@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df
data = download_data('RELIANCE.NS', "2000-01-01", "2022-12-31")
scaler = StandardScaler()

Quarterlydata = data.resample('Q').mean()
Monthlydata = data.resample('M').mean()





############## Dashboard #################
def dashboard():
    
    st.subheader("Data Information")


    radio_value = st.selectbox("Column Names",data.columns.tolist())
    if radio_value == "Open":
        selected_col = "Open"
    elif radio_value == "High":
        selected_col = "High"
    elif radio_value == "Low":
        selected_col =("Low") 
    elif radio_value == "Close":
        selected_col =("Close") 
    elif radio_value == "Adj Close":
        selected_col =("Adj Close") 
    elif radio_value == "Volume":
        selected_col =("Volume") 


######## Data Description Area ##############
    columns = st.columns(5)

    with columns[0]:
        st.write("Count")
        st.success(int(data[selected_col].count()))

    with columns[1]:
        st.write("Mean")
        st.warning(int(data[selected_col].values.mean()))
        
    with columns[2]:
        st.write("Standard Deviation")
        st.info(int(data[selected_col].values.std()))
        
    with columns[3]:
        st.write("Min")
        st.error(int(data[selected_col].values.min()))
        
    with columns[4]:
        st.write("Max")
        st.success(int(data[selected_col].values.max()))

######### Dashboard Chart Area  ############

    dash_chart_columns = st.columns(2)

    with dash_chart_columns[0]:
        # st.write(Quarterlydata.describe())
        fig_1 = px.area(data, x=data.index, y=selected_col, title='Stock price analysis')
        fig_1.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
        )
        st.plotly_chart(fig_1)

    with dash_chart_columns[1]:
        fig_2 = go.Figure()
        # Add the closing prices trace
        fig_2.add_trace(go.Scatter(
            x=data.index,
            y=data[selected_col],
            name=selected_col
        ))

        # Add the moving average trace
        fig_2.add_trace(go.Scatter(
            x=data.index,
            y=data[selected_col].rolling(50).mean(),
            name='Moving Average',
            line=dict(color='green')
        ))

        # Set the layout
        fig_2.update_layout(
            title='Closing Prices with Moving Average',
            xaxis_title='Date',
            yaxis_title=selected_col
        )
        st.plotly_chart(fig_2)
    
    
    st.dataframe(data.describe(),use_container_width=True)



############## Data Info #################
def data_Visualize():
    st.subheader('Technical Indicators')
    option = st.selectbox('Make a choice', ['Adj Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Adj Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Adj Close':
        st.write('Adj Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
        st.write('Bollinger Bands are a technical analysis tool that helps traders identify overbought and oversold conditions in a security. They are composed of three lines: a **simple moving average (SMA)**, an **upper band**, and a **lower band**. The upper band is typically two standard deviations above the SMA, and the lower band is typically two standard deviations below the SMA.')
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
        st.write('The Relative Strength Indicator (RSI) is a popular technical analysis tool used to measure the momentum of a financial instrument, such as a stock or an index. It helps traders and analysts identify potential overbought or oversold conditions in the market.The RSI is calculated based on the average gains and losses over a specified period of time. The formula for calculating the RSI is as follows: )')
        st.code('RSI = 100 - (100 / (1 + RS)')
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Expoenetial Moving Average')
        st.line_chart(ema)



############## MAPE Calculation Function ###################
def MAPE(pred,org): #mean absolute percentage error
    temp = np.abs((pred-org)/org)*100 #mape function = prediction value-orginal value for error(e),by orginal value *100 for percentage(p),.abs for absolute
    return np.mean(temp)  #and taking mean of it toget MAPE

############## RMSE Calculation Function ###################
def RMSE(org, pred):
    rmse=np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
    return rmse

############## predict #######################################
def predict():
    st.subheader("Prediction")
    
    from pmdarima import auto_arima

    # Ignore harmless warnings
    import warnings
    warnings.filterwarnings("ignore")

    # st.dataframe(Quarterlydata)
    # # Fit auto_arima function to Reliance dataset
    # stepwise_fit = auto_arima(Quarterlydata['Adj Close'], start_p = 1, start_q = 1, #saying start the p and q values with 1 respectively
    #                         max_p = 3, max_q = 3, m = 12, #end with 3 
    #                         start_P = 0, seasonal = True, #also check seasonality 
    #                         d = None, D = 1, trace = True, #integrated i.e differency as 1
    #                         error_action ='ignore', # we don't want to know if an order does not work
    #                         suppress_warnings = True, # we don't want convergence warnings
    #                         stepwise = True) # set to stepwise

    # # To print the summary
    # stepwise_fit.summary()

    Arima = ARIMA(Quarterlydata['Adj Close'], order=(2,1,2))
    
    result = Arima.fit()
    # st.write(result.summary())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Quarterlydata.index, y=Quarterlydata['Adj Close'], mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=Quarterlydata.index, y=result.fittedvalues, mode='lines', name='Predicted', line=dict(color='green')))
    fig.update_layout(showlegend=True, legend=dict(orientation="h")  )
    st.plotly_chart(fig, use_container_width=True)

    rmse_ARIMA = RMSE(Quarterlydata['Adj Close'], result.fittedvalues)
    mape_ARIMA = MAPE(result.fittedvalues,Quarterlydata['Adj Close'])
    st.write("MAPE Value for ARIMA Model = " ,mape_ARIMA)
    st.write("RMSE Value for ARIMA Model = " ,rmse_ARIMA)

    # st.dataframe(Quarterlydata)

    # Spliting data into train / test sets
    train = Quarterlydata.iloc[:len(Quarterlydata)-12]
    test = Quarterlydata.iloc[len(Quarterlydata)-12:] # Taking one year(12 months) for testing
    
    # Fitting SARIMAX(2, 1, 2)x(0, 1, 0, 12) on the training set
    from statsmodels.tsa.statespace.sarimax import SARIMAX #SARIMAX means Seasonal ARIMA with exogenous variables
    
    model = SARIMAX(train['Adj Close'], 
                    order = (2,1,2), 
                    seasonal_order =(0, 1, 0, 12))
    
    result = model.fit()
    result.summary()

    start = len(train)
    end = len(train) + len(test) - 1
    
    # Predictions for one-year against the test set
    predictions = result.predict(start, end,
                                typ = 'levels').rename("Predictions")
    
    # plot predictions and actual values
    predictions.plot(legend = True)
    test['Adj Close'].plot(legend = True)

    # rmse_ARIMA = RMSE(test.Adj_Close, predictions)
    # mape_ARIMA = MAPE(predictions,test.Adj Close)
    # print("MAPE Value for ARIMA Model = " ,mape_ARIMA)
    # print("RMSE Value for ARIMA Model = " ,rmse_ARIMA)

    # Train the model on the full dataset
    model = model = SARIMAX(Quarterlydata['Adj Close'], 
                            order = (2,1,2), 
                            seasonal_order =(0, 1, 0, 12))
    result = model.fit()
    
    # Forecast for the next 2 years
    forecast = result.predict(start = len(Quarterlydata), 
                            end = (len(Quarterlydata)-1) + 2 * 12, 
                            typ = 'levels').rename('Forecast')
    
    # Plot the forecast values
    # Quarterlydata['Adj Close'].plot(figsize = (12, 5), legend = True)
    # forecast.plot(legend = True)

   # Create a trace for the historical data
    trace1 = go.Scatter(
        x=Quarterlydata.index,
        y=Quarterlydata['Adj Close'],
        mode='lines',
        name='Historical Data'
    )

    # Create a trace for the forecast data
    trace2 = go.Scatter(
        x=forecast.index,
        y=forecast,
        mode='lines',
        name='Forecast'
    )

    # Create the layout for the plot
    layout = go.Layout(
        title='Quarterly Data with Forecast',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Adj Close'),
        legend=dict(orientation="h")
    )

    # Combine the traces and layout into a figure
    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)

    # Display the plot
    st.plotly_chart(fig,use_container_width=True)



################### Topbar #######################
cols = st.columns(2)
with cols[0]:
    st.header(':chart_with_upwards_trend: :blue[Reliance] Stock Price Prediction App')

with cols[1]:
    option = st.selectbox('Make a choice', ['Dashboard','Visualization', 'Prediction'])

if option == 'Visualization':
    data_Visualize()

elif option == 'Dashboard':
    dashboard()
else:
    predict()
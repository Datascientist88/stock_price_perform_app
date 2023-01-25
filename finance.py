import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st 
import yfinance as yf
import plotly_express as px
import plotly.graph_objects as go
import datetime 
import datetime as dt 
from streamlit_lottie import st_lottie
import json
import requests
from prophet import Prophet 
from prophet.plot import plot_plotly
from stocknews import StockNews
from pandas_datareader import data as pdr
# the application layout 
st.set_page_config('stock forecasting Dashboard',layout='wide')
# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(
     """
            <style>
            .main {
                background_color: #0E1117; 
            }
            </style>
            """ , unsafe_allow_html=True)

st.sidebar.title('STOCK PRICE PERFORMANCE')
url="https://assets6.lottiefiles.com/packages/lf20_3kjzsbjv.json"
r=requests.get(url)
Lottiecode=None if r.status_code !=200 else r.json()
try:
    with st.sidebar:
        st_lottie(Lottiecode,height=200,width=200,speed=1, loop=True)
except:
    pass
def nearest_business_day(Date:datetime.date):
    if Date.weekday()==5:
        Date=Date-datetime.timedelta(days=1)
    if Date.weekday()==6:
        Date=Date+datetime.timedelta(days=1)
    return Date
yesterday=datetime.date.today()-datetime.timedelta(days=1)
yesterday=nearest_business_day(yesterday)
# setting the widgets:
default_start=yesterday-datetime.timedelta(days=700)
default_start=nearest_business_day(default_start)
## NEW APPLICATION LAYOUT 
ticker =st.sidebar.text_input('ENTER A VALID STOCK TICKER SYMBOL:','GOOG')
selection=st.sidebar.container()
selection.markdown('Forecasting Stock')
sub_columns=selection.columns(2)
start_date=sub_columns[0].date_input('FROM',value=default_start,max_value=yesterday-datetime.timedelta(days=1))
start_date=nearest_business_day(start_date)
end_date=sub_columns[1].date_input('TO')
# design tabs for current stock performance and FUture forecast 
current_stock_price,Future_forecast=st.tabs(['CURRENT STOCK PRICE','FUTURE FORECAST'])
with current_stock_price:
    data=yf.download(ticker,start=start_date,end=end_date)
    fig=go.Figure()
    fig = fig.add_trace( go.Scatter( x=data.index, y=data['Adj Close'],mode="lines",marker=dict(color='rgb(102,255,0)', )) )
    fig.update_layout(width=1300,margin=dict(l=0, r=1, t=0, b=0, pad=0), legend=dict(x=0, y=0.99,traceorder="normal", font=dict(size=12),),autosize=False,template="plotly_dark", xaxis_rangeslider_visible=True,)
    st.plotly_chart(fig)
    pricing_data ,news=st.tabs(['Pricing Data','Top 10 News'])
    with pricing_data:
        st.header("Price Movements")
        delta=data
        delta['% Change']=round(data['Adj Close']/data['Adj Close'].shift(1)-1,3)
        delta.dropna(inplace=True)
        annual_return=round(delta['% Change'].mean()*252*100,3)
        stdv=np.std(delta['% Change'])*np.sqrt(252)
        h1 ,h2,h3=st.columns(3)
        with h1:
            st.write(f"The Annual Return of {ticker } is :", annual_return ,'%')
        with h2:
            st.write(f"The Standard Deviation of  {ticker} is" ,round(stdv*100,2) ,'%')
        with h3:
            st.write(f" Risk Adjusted Return for {ticker} is",annual_return/ round(stdv*100,2),'%')
        seg1,seg2=st.columns(2)
        with seg1:
           st.write(delta)
        with seg2:
            fig3=go.Figure()
            st.write(f"Trading Volume of {ticker}")
            fig3 = fig3.add_trace( go.Scatter( x=data.index, y=data['Volume'],mode="lines",marker=dict(color='rgb(255,16,240)', )) )
            fig3.update_layout(width=700,margin=dict(l=0, r=1, t=0, b=0, pad=0), legend=dict(x=0, y=0.99,traceorder="normal", font=dict(size=12),),autosize=False,template="plotly_dark", xaxis_rangeslider_visible=True,)
            st.plotly_chart(fig3)
    with news:
        st.header(f"NEWS OF {ticker}")
        sn=StockNews(ticker,save_news=False)
        df_news=sn.read_rss()
        for i in range(10):
            st.subheader(f"HEADLINE {i+1}")
            st.write(df_news['published'][i])
            st.write(df_news['title'][i])
            st.write(df_news['summary'][i])
            title_sentiment=df_news['sentiment_title'][i]
            st.write(f" Title Sentiment{ title_sentiment}")
            news_sentiment=df_news['sentiment_summary'][i]
            st.write(f" News Sentiment{ news_sentiment}")
with Future_forecast:
    START=start_date
    TODAY=datetime.date.today().strftime("%Y-%m-%d")
    data2=yf.download(ticker,START,TODAY)
    p1,p2=st.columns(2)
    # forecasting the price
    end= dt.datetime.today()
    yf.pdr_override()
    df=pdr.get_data_yahoo(ticker,start_date,end)
    df_modified=df.reset_index()
    df_train=df_modified[['Date','Close']]
    df_train=df_train.rename(columns={"Date":"ds","Close":"y"})
    df_train['ds']=pd.to_datetime(df_train['ds'],format="%Y-%m-%d")
    df_train['ds']=df_train['ds'].dt.tz_convert(None)
    m=Prophet()
    m.fit(df_train)
    
    with p1:
        n_years=st.slider("Years Of Prediction:" ,1,4)
        period=n_years*365
        future=m.make_future_dataframe(periods=period)
        forecast=m.predict(future)
        fig2=plot_plotly(m,forecast)
        st.write(f" Forecasting for stock {ticker}")
        st.plotly_chart(fig2)
    with p1 :
        fig1=go.Figure()
        fig1.add_trace(go.Scatter(x=data2.index,y=data2['Close'],name="Stock_Close"))
        fig1.update_layout(title_text=f"TIME SERIES STOCK DATA FOR {ticker}",xaxis_rangeslider_visible=True,template="plotly_dark",width=900,margin=dict(l=0, r=1, t=0, b=0, pad=0), legend=dict(x=0, y=0.99,traceorder="normal", font=dict(size=12),))
        st.write(f" Actual Price for stock {ticker}")
        st.plotly_chart(fig1)
    

    







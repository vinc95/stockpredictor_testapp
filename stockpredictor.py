import streamlit as st
import yfinance as yf
from datetime import date
from fbprophet import Prophet

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Web App")

st.sidebar.write("""
This is a mini web app demo project using python libraries such as Streamlit, fbprophet etc
""")

st.sidebar.write ("This stock price predictor used the data from 2015-01-01 until today as a reference to predict the stock price for the future.")

st.sidebar.write ("For more info, please contact:")

st.sidebar.write("<a href='https://www.linkedin.com/in/jiajunlok/'>Lok Jia Jun </a>", unsafe_allow_html=True)

stocks = ("AAPL","AMZN","FB","GME","GOOG","MSFT","NIO","TSLA")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 10)
period = n_years * 365

#Plotting
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Data is loading...")
data = load_data(selected_stock)

st.subheader('Raw data')
st.write(data.tail())

#Forecasting
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

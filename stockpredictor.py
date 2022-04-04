import streamlit as st
import yfinance as yf
from datetime import date
from fbprophet import Prophet

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Web App")

stocks = ("0208.KL","AAPL","GOOG","FB","MSFT","GME")
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
#data_load_state.text("Loading Data... Taadaa... DONE!")


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

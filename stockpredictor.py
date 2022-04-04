import streamlit as st
import yfinance as yf
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly 
from matplotlib import pyplot as plt

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Web App")

stocks = ("AAPL","GOOG","FB","MSFT","GME")
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
data_load_state.text("Loading Data... Taadaa... DONE!")


st.subheader('Raw data')
st.write(data.tail())


#def plot_raw_data():
#    fig = plt.Figure()
#    plt.scatter(x=data['Date'], y=data['Open'], c='b', marker='x', label='stock_open')
#    plt.scatter(x=data['Date'], y=data['Close'], c='r', marker='s', label='close')
#    ###fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
#    st.plotly_chart(fig)

# plot_raw_data()


#Forecasting
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(Forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.pyplot_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

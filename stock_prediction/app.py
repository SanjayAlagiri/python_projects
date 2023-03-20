import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

st.title('stock predicter')

user_input = st.text_input('enter stock ticker' , 'KPITTECH.NS')
yf.pdr_override()
df = yf.download(user_input, '2010-01-01', '2019-12-31',  'yahoo')

#describing data
st.subheader('data from 2019-2022')
st.write(df.describe())

#visualization

st.subheader('Closing price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs time chart with ma100')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs time chart with ma200')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

#splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)  

model = load_model('keras_model.h5')

#Testing part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (100 , input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i , 0])

x_test , y_test = np.array(x_test) , np.array(y_test)
y_prediction = model.predict(x_test)


scale_factor = 1/0.00200562
y_prediction = y_prediction * scale_factor
y_test = y_test * scale_factor

#final graph
st.subheader('prediction vs original')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test , 'b' , label = 'original price')
plt.plot(y_prediction , 'r' , label = 'predict price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)
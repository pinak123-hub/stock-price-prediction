import tensorflow as tf
# tensorflow==2.6.0
import numpy as np
# numpy==1.19.5
import pandas as pd
# pandas==1.2.4
import matplotlib.pyplot as plt
# matplotlib=3.1.3
import pandas_datareader as data
# pandas-datareader==0.10.0
from keras.models import load_model
# keras==2.6.0
import streamlit as st
# streamlit==1.7.0


start="2011-01-01"
end="2022-04-26"


st.title("Stock Price Prediction")
user_comp=st.text_input("Enter Company Name","AXISBANK.NS")
df=data.DataReader(user_comp,"yahoo",start,end)


st.subheader("Data from 2011 to 2022")
st.write(df.describe())

st.write(f"{user_comp} company chart")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.write(f"{user_comp} company 100 days average price")
avg100=df.Close.rolling(100).mean()
fig2=plt.figure(figsize=(12,6))
plt.plot(avg100)
plt.plot(df.Close)
st.pyplot(fig2)

st.write(f"{user_comp} company 100 days and 200 days average price")
avg100=df.Close.rolling(100).mean()
avg200=df.Close.rolling(200).mean()
fig3=plt.figure(figsize=(12,6))
# plt.figure(figsize=(12,6))
plt.plot(avg100,"r")
plt.plot(avg200,"o")
plt.plot(df.Close,"b")
st.pyplot(fig3)


training_data=pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
testing_data=pd.DataFrame(df["Close"][int(len(df)*0.70):])

from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler(feature_range=(0,1))
training_data_arr=minmax.fit_transform(training_data)



model=load_model("model_lstm.h5")
past_100=training_data.tail(100)
final_df=past_100.append(testing_data,ignore_index=True)
ip_data=minmax.fit_transform(final_df)


x_test=[]
y_test=[]
for i in range(100,ip_data.shape[0]):
  x_test.append(ip_data[i-100:i])
  y_test.append(ip_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)

scaler_=minmax.scale_

scale_fac=1/scaler_[0]

y_predicted=y_predicted*scale_fac
y_test=y_test*scale_fac


st.subheader(f"Predicted Price of {user_comp}")
fig4=plt.figure(figsize=(12,6))
plt.plot(y_test,"b",label="origianl Price")
plt.plot(y_predicted,"r",label="predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)


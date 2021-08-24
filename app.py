#Importações básicas

import pandas as pd
import numpy as np
import math
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image
import datetime as dt
from plotly import graph_objects as go
import streamlit as st

st.title('Previsor de tendências de ações da Bolsa de Valores')

img = Image.open('img.jpg')
st.image(img)

# Usuário digita o nome do ativo
user_input = st.text_input('Selecione o ativo desejado:', 'FB')

#Carregando a base de dados
df = web.DataReader(user_input, data_source='yahoo', start = '2012-01-01', end = dt.datetime.now().strftime('%Y-%m-%d'))
df.reset_index(inplace = True)
#Descrevendo os dados para o usuário

st.subheader('Dados de 2012 - 2021')
st.write(df.describe())

#Criando um df somente com a coluna 'Close'
#df_close = df.filter(['Close'])

#Visualizações

st.subheader('Preço do fechamento vs tempo')
def plot_graf():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'], y = df['Close'], name = 'Fechamento dos Ativos'))
    fig.layout.update(title_text = 'Preço do Fechamento vs Tempo', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
plot_graf()
#fig = plt.figure(figsize=(12,6))
#plt.plot(df_close)
#st.pyplot(fig)

# Média móvel de 60 dias

st.subheader('Preço do fechamento vs tempo e a média móvel de 60 dias')

def plot_mm60():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'], y = df['Close'].rolling(60).mean(), name = 'Média movel de 60 dias'))
    fig.add_trace(go.Scatter(x = df['Date'], y = df['Close'], name = 'Fechamento dos Ativos'))
    fig.layout.update(title_text = 'Média móvel de 60 dias', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
plot_mm60()

#fig = plt.figure(figsize=(16,8))
#plt.plot(df_close)
#plt.plot(med_mov_60)
#st.pyplot(fig)

# Gráfico da média móvel de 60 e 120 dias

st.subheader('Gráfico da média móvel de 60 e 150 dias')

def plot_mm150():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'], y = df['Close'].rolling(60).mean(), name = 'Média movel de 60 dias'))
    fig.add_trace(go.Scatter(x = df['Date'], y = df['Close'].rolling(150).mean(), name = 'Média movel de 150 dias'))
    fig.add_trace(go.Scatter(x = df['Date'], y = df['Close'], name = 'Fechamento dos Ativos'))
    fig.layout.update(title_text = 'Média móvel de 60 e 150 dias', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
plot_mm150()

df_close = df.filter(['Close'])
#Convertendo para um array
dataset = df_close.values

#Tamanho do conjunto de treino
df_train_len = math.ceil(len(dataset)*0.8)

#Escalonando os dados

scaler = MinMaxScaler(feature_range=(0,1))
scaled_df = scaler.fit_transform(df_close)

# Criando o df de treino
train_data = scaled_df[0:df_train_len:,]

#dividindo entre x_train e Y_train

x_train = []
y_train = []

for i in range(60, len(train_data)): 
    x_train.append(train_data[i-60:i,0]) # vai do 0 a posição 59
    y_train.append(train_data[i,0])  # valor da posição 60 (valor a ser previsto)
    
x_train, y_train = np.array(x_train), np.array(y_train)

# Passando para 3d (LSTM model)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Carregando o modelo salvo

model = load_model('awari_modelo.h5')


#Criando o dataset de teste
test_data = scaled_df[df_train_len - 60:, :]

#Criando x_test e y_test
x_test = []
y_test = dataset[df_train_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


#Convertendo para um array
x_test = np.array(x_test)

#Reshape para 3d
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Obter os predicted values
predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)

# Plotando os dados
st.subheader('Gráficos dos valores reais e preditos')

train = df_close[:df_train_len]
valid = df_close[df_train_len:]
valid['Predictions'] = predictions



fig2 = plt.figure(figsize = (20,12))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Closing Price USD')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc = 'lower right')
st.pyplot(fig2)

# Mostrando os valores reais e os previstos

st.subheader('Valores reais e valores previstos')
st.dataframe(valid.style.format("{:.4f}"))
#st.write(valid)
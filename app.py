import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go

# Tabela de tradução dos meses
month_translation = {
    "January": "Janeiro", "February": "Fevereiro", "March": "Março",
    "April": "Abril", "May": "Maio", "June": "Junho",
    "July": "Julho", "August": "Agosto", "September": "Setembro",
    "October": "Outubro", "November": "Novembro", "December": "Dezembro"
}

# Função para traduzir meses
def translate_month(date):
    month = date.strftime('%B')  # Nome do mês em inglês
    return month_translation.get(month, month) + date.strftime(' de %Y')

# Criação do DataFrame com dados simulados
data = {
    "Data": [
        "2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01", "2021-10-01", "2021-11-01", "2021-12-01",
        "2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01", "2022-06-01", "2022-07-01",
        "2022-08-01", "2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01", "2023-01-01", "2023-02-01",
        "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01",
        "2023-10-01", "2023-11-01", "2023-12-01", "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01",
        "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01", "2024-10-01", "2024-11-01"
    ],
    "Acervo Líquido": [
        128273, 127646, 128593, 130399, 134408, 136226, 133909, 137246, 138903, 143337, 144011, 141838, 138134,
        140012, 135678, 133136, 129895, 128681, 124008, 124396, 122481, 123484, 122739, 121646, 121419, 120855,
        116857, 116075, 110279, 108270, 105288, 103891, 99075, 96698, 94320, 91300, 87147, 85509, 80484, 79136,
        78472, 74670
    ],
    "Julgamentos": [
        15865, 17767, 18126, 15152, 15755, 18342, 12574, 6908, 16342, 20303, 18771, 22260, 22772, 19539, 25820,
        21717, 19793, 21357, 16348, 9720, 18773, 21131, 18317, 23992, 22255, 21722, 28074, 21283, 25173, 24021,
        17149, 12267, 24319, 21258, 25502, 23240, 24444, 25194, 26736, 23724, 24551, 23244
    ],
    "Saldo de Entradas": [
        17753, 17365, 18103, 16760, 19821, 19348, 10078, 10720, 17376, 24818, 19650, 19660, 19072, 21981, 21494,
        19221, 16608, 20308, 11906, 10146, 17042, 22458, 17386, 22301, 22078, 21185, 23906, 20051, 19444, 22048,
        14796, 10956, 19739, 19046, 23512, 20710, 20440, 23236, 21525, 22294, 22175, 22262
    ]
}

# Criação do DataFrame
df = pd.DataFrame(data)

# Conversão e indexação por data
df['Data'] = pd.to_datetime(df['Data'])
df.set_index('Data', inplace=True)

################

# Função para criar e ajustar o modelo SARIMAX
def sarimax_model(endog, exog=None, steps=12):
    model = SARIMAX(endog, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=steps, exog=exog.iloc[-steps:])
    forecast_index = pd.date_range(endog.index[-1], periods=steps+1, freq='M')[1:]
    forecast_values = forecast.predicted_mean
    return model_fit, forecast_values, forecast_index

# Função para decomposição de séries temporais
def decompose_series(series):
    decomposition = seasonal_decompose(series, model='additive', period=12)
    decomposition.trend = decomposition.trend.interpolate(method='linear')
    decomposition.trend = decomposition.trend.reindex(series.index)
    return decomposition

# Funções para gráficos

def plot_series(actual, predicted, index, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode='lines', name='Observado'))
    fig.add_trace(go.Scatter(x=index, y=predicted, mode='lines', name='Predito', line=dict(color='red')))
    fig.update_layout(title=title, xaxis_title='Data', yaxis_title='Valor', legend_title='Séries')
    st.plotly_chart(fig)

def plot_decomposition(decomposition, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Tendência'))
    fig.update_layout(title=title, xaxis_title='Data', yaxis_title='Tendência')
    st.plotly_chart(fig)

# Interface do Streamlit
st.markdown("<h1 style='text-align: center;'>NEAD WebApp - Antecipação de Cenários Futuros</h1>", unsafe_allow_html=True)
st.write('Análise de séries temporais e predições baseadas em machine learning')

# Seleção da variável a ser predita
option = st.selectbox('Escolha a métrica para realizar a predição:', ('Acervo Líquido', 'Saldo de Entradas', 'Julgamentos'))

# Selecionar o número de meses para predição
steps = st.slider('Número de meses para predição', 1, 12, 6)

# Definição de endógena e exógena
if option == 'Saldo de Entradas':
    endog = df['Saldo de Entradas']
    exog = df[['Julgamentos']]
elif option == 'Julgamentos':
    endog = df['Julgamentos']
    exog = df[['Saldo de Entradas']]
else:
    endog = df['Acervo Líquido']
    exog = df[['Saldo de Entradas', 'Julgamentos']]

# Ajustar o modelo e obter a previsão
model_fit, forecast_values, forecast_index = sarimax_model(endog, exog, steps)

# Decomposição da série temporal
decomposition = decompose_series(endog)

# Gráfico da Tendência
st.subheader('Componente de Tendência')
plot_decomposition(decomposition, 'Tendência')

# Gráfico da série temporal com predições
st.subheader(f'Série Temporal e Valores Preditores - {option}')
plot_series(endog, forecast_values, forecast_index, f'Predição para {option}')

# Mostrar AIC do modelo
st.write(f"AIC do modelo de predição (SARIMAX): {model_fit.aic:.2f}")

# Últimos valores observados no mesmo mês do ano anterior
last_year_values = endog.shift(12).reindex(forecast_index)

# Resultados da predição
results = pd.DataFrame({
    'Mês e Ano': forecast_index.strftime('%B de %Y'),
    'Valor Predito': forecast_values,
    'Último Valor Observado Previamente': last_year_values.values,
    'Variação (%)': (forecast_values - last_year_values.values) / last_year_values.values * 100
})

# Exibir a tabela com valores preditos
st.write('Tabela de Predições:')
st.dataframe(results)

# Exibir uma tabela estática
st.write('Verificação da predição em ciclo mensal anterior')
static_data = {
    'Métricas de Produção': ['Julgamentos', 'Casos Novos', 'Acervo Líquido'],
    'Observado - Novembro de 2024': ['23.244', '22.262', '74.670'],
    'Predito - Novembro de 2024': ['23.813', '20.544', '73.986'],
    'Diferença absoluta': ['-569', '1.718', '684'],
    'Diferença relativa(%)': ['2,4%', '-7,7%', '-0,9%']
}
static_df = pd.DataFrame(static_data)
st.table(static_df)

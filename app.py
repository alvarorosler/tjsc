import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Criação do DataFrame 2G e variáveis exógenas de apoio do 1G e JE
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
    ],
    "Sentenças_JE": [
        18863, 19856, 22137, 22173, 19551, 21065, 14214, 11987, 18521, 22727, 20240, 24328, 24043, 23402, 26644,
        25914, 22816, 24518, 15195, 14465, 24928, 30768, 22134, 29769, 30610, 26651, 29341, 27624, 29727, 28087,
        16360, 15832, 23838, 25241, 27921, 27249, 28474, 32874, 29293, 30977, 31149, 26814
    ],
    "Sentenças_1G": [
        53502, 58446, 61225, 57419, 54986, 62962, 63021, 47877, 68128, 79135, 64425, 69594, 68537, 71062, 73181,
        69477, 67641, 77003, 75801, 55216, 65747, 79264, 64102, 78936, 73324, 73785, 79024, 75082, 74900, 74220,
        74222, 72113, 72503, 80012, 78849, 73568, 73559, 81543, 79823, 77201, 95247, 76834
    ]
}

df = pd.DataFrame(data)

# Converter a coluna 'Data' para o tipo datetime
df['Data'] = pd.to_datetime(df['Data'])

# Definir a coluna 'Data' como índice
df.set_index('Data', inplace=True)

################

# Função para criar e ajustar o modelo SARIMAX
def sarimax_model(endog, exog=None, steps=12):
    # SARIMAX(1,1,1)(1,1,1)[12]
    model = SARIMAX(endog, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=steps, exog=exog[-steps:])
    forecast_index = pd.date_range(endog.index[-1], periods=steps+1, freq='M')[1:]
    forecast_values = forecast.predicted_mean
    return model_fit, forecast_values, forecast_index

# Função para decomposição de séries temporais
def decompose_series(series):
    decomposition = seasonal_decompose(series, model='additive', period=12)
    return decomposition

# Função para criar gráficos de série temporal e decomposição
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

def plot_seasonality(decomposition, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Sazonalidade'))
    fig.update_layout(title=title, xaxis_title='Data', yaxis_title='Sazonalidade')
    st.plotly_chart(fig)

# Função para plotar o índice de sazonalidade mensal em relação ao desvio percentual da média anual
def plot_seasonal_index(series, title):
    # Cálculo da média mensal
    monthly_avg = series.groupby(series.index.month).mean()

    # Cálculo da média anual
    annual_avg = series.mean()

    # Cálculo do desvio percentual em relação à média anual
    seasonal_index = (monthly_avg - annual_avg) / annual_avg * 100

    # Criando o gráfico com Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_avg.index, y=seasonal_index.values, name='Índice de Sazonalidade (%)'))
    fig.update_layout(title=title, xaxis_title='Mês', yaxis_title='Desvio Percentual da Média Anual (%)')
    st.plotly_chart(fig)

# Interface do Streamlit
st.markdown("<h1 style='text-align: center;'>NEAD WebApp - Antecipação de Cenários Futuros para o 2G</h1>", unsafe_allow_html=True)
st.write('Análise de séries temporais e predições baseadas em machine learning')
st.write('Versão: 1.5')
st.write('Data: 11/12/2024')
st.write('Cientista de Dados: Álvaro Rösler')
st.write('Supervisão: Sérgio Weber')

# Seleção da variável a ser predita
option = st.selectbox('Escolha a métrica para realizar a predição:', ('Acervo Líquido', 'Saldo de Entradas', 'Julgamentos'))

# Selecionar o número de meses para predição
steps = st.slider('Número de meses para predição', 1, 12, 6)

# Variável alvo
if option == 'Saldo de Entradas':
    endog = df['Saldo de Entradas']
    exog = df[['Sentenças_JE', 'Sentenças_1G']]
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

# Gráfico de Sazonalidade
st.subheader('Componente de Sazonalidade')
plot_seasonality(decomposition, 'Sazonalidade')

# Gráfico do Índice de Sazonalidade Mensal (Desvio Percentual da Média Anual)
st.subheader('Índice de Sazonalidade Mensal (Desvio Percentual da Média Anual)')
plot_seasonal_index(endog, 'Índice de Sazonalidade Mensal (%)')

# Formatar a coluna 'Data' para exibir o mês e ano no formato desejado (Outubro de 2024, etc.)
results = pd.DataFrame({
    'Mês e Ano': forecast_index.strftime('%B de %Y'),  # Agora os meses estão em português
    'Valor Predito': forecast_values.apply(lambda x: f"{x:,.0f}".replace(",", ".")),
    'Último Valor Observado Previamente': endog[-steps:].values,
    'Variação (%)': (forecast_values - endog[-steps:].values) / endog[-steps:].values * 100
})

# Gráfico da série temporal com predições
st.subheader(f'Série Temporal e Valores Preditores - {option}')
plot_series(endog, forecast_values, forecast_index, f'Predição para {option}')

# Mostrar AIC do modelo
st.write(f"AIC do modelo de predição (SARIMAX): {model_fit.aic:.0f}")

# Exibir a tabela com valores preditos, observados e variação percentual, ocultando a coluna extra de data
st.write('Tabela de Predições:')
st.dataframe(results)

# Exibir uma tabela estática
st.subheader('Resumo Estático de Predições - Novembro de 2024')
static_data = {
    'Métrica': ['Julgamentos', 'Casos Novos', 'Acervo Líquido'],
    'Observado Nov/2024': ['23.244', '22.262', '74.670'],
    'Predito Nov/2024': ['23.813', '20.544', '73.986'],
    'Dif. Abs.': ['-569', '1.718', '684'],
    'Dif. %': ['2,4%', '-7,7%', '-0,9%']
}
static_df = pd.DataFrame(static_data)
st.table(static_df)

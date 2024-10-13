import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_stock_data(ticker, start_date, end_date):
    ticker = ticker.upper()  # Converte para maiúsculas
    if not ticker.endswith('.SA'):
        ticker += '.SA'
    
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"Não foi possível obter dados para o ticker {ticker}")
    return data

def calculate_percentage_change(current_price, target_price):
    return ((target_price - current_price) / current_price) * 100

def calculate_volatility(data):
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    return volatility

def monte_carlo_simulation(current_price, volatility, days, num_simulations=10000):
    dt = 1/252  # Assumindo 252 dias de negociação por ano
    mu = 0  # Assumindo retorno médio diário de 0 para simplificar
    
    # Gerando caminhos aleatórios
    random_walks = np.exp(
        (mu - 0.5 * volatility**2) * dt +
        volatility * np.sqrt(dt) * np.random.normal(size=(num_simulations, days))
    )
    
    # Calculando os caminhos de preço
    price_paths = current_price * np.cumprod(random_walks, axis=1)
    
    return price_paths

def main():
    st.title("Análise de Ações Brasileiras com Simulação de Monte Carlo")

    ticker = st.text_input("Digite o ticker da ação brasileira (ex: PETR4, VALE3):").upper()  # Converte para maiúsculas imediatamente
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    start_date = st.date_input("Data inicial", value=start_date)
    end_date = st.date_input("Data final", value=end_date)

    target_price_1 = st.number_input("Digite o primeiro preço-alvo (R$):", min_value=0.01, step=0.01)
    target_price_2 = st.number_input("Digite o segundo preço-alvo (R$):", min_value=0.01, step=0.01)

    if ticker and target_price_1 and target_price_2 and start_date < end_date:
        try:
            data = get_stock_data(ticker, start_date, end_date)
            current_price = data['Close'].iloc[-1]
            lowest_price = data['Close'].min()
            highest_price = data['Close'].max()

            st.write(f"Preço atual de {ticker}: R${current_price:.2f}")

            change_to_target_1 = calculate_percentage_change(current_price, target_price_1)
            change_to_target_2 = calculate_percentage_change(current_price, target_price_2)
            change_from_lowest = calculate_percentage_change(lowest_price, current_price)
            change_to_highest = calculate_percentage_change(current_price, highest_price)

            volatility = calculate_volatility(data)

            # Monte Carlo Simulation para 30 dias
            days_30 = 30
            num_simulations = 10000
            mc_simulations = monte_carlo_simulation(current_price, volatility, days_30, num_simulations)
            
            prob_mc_target_1 = np.mean(mc_simulations[:, -1] >= target_price_1)
            prob_mc_target_2 = np.mean(mc_simulations[:, -1] >= target_price_2)

            # Exibição dos resultados
            st.subheader("Análise de Probabilidades (30 dias)")
            st.write(f"Volatilidade anualizada: {volatility*100:.2f}%")
            st.write(f"Probabilidade de atingir Alvo 1 (R${target_price_1:.2f}) em 30 dias: {prob_mc_target_1*100:.2f}%")
            st.write(f"Probabilidade de atingir Alvo 2 (R${target_price_2:.2f}) em 30 dias: {prob_mc_target_2*100:.2f}%")

            # Gráfico de barras
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=['Menor Preço', 'Preço Atual', 'Maior Preço', 'Alvo 1', 'Alvo 2'],
                y=[lowest_price, current_price, highest_price, target_price_1, target_price_2],
                text=[
                    f'R${lowest_price:.2f}<br>({change_from_lowest:.2f}%)',
                    f'R${current_price:.2f}',
                    f'R${highest_price:.2f}<br>({change_to_highest:.2f}%)',
                    f'R${target_price_1:.2f}<br>({change_to_target_1:.2f}%)',
                    f'R${target_price_2:.2f}<br>({change_to_target_2:.2f}%)'
                ],
                textposition='auto',
                marker_color=['purple', 'blue', 'orange', 'red', 'green']
            ))

            fig.update_layout(
                title=f'Comparação de Preços para {ticker}',
                yaxis_title='Preço (R$)',
                showlegend=False
            )

            st.plotly_chart(fig)

            # Gráfico histórico com algumas simulações de Monte Carlo
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Preço Histórico'))
            
            # Adicionando algumas simulações ao gráfico
            dates_future = pd.date_range(start=data.index[-1], periods=days_30+1, freq='B')[1:]
            for i in range(min(50, num_simulations)):  # Plotando 50 simulações ou menos
                fig_hist.add_trace(go.Scatter(x=dates_future, y=mc_simulations[i], mode='lines', 
                                              opacity=0.1, line=dict(color='gray'), showlegend=False))

            fig_hist.add_hline(y=current_price, line_dash="dash", line_color="blue", annotation_text="Preço Atual")
            fig_hist.add_hline(y=target_price_1, line_dash="dash", line_color="red", annotation_text="Alvo 1")
            fig_hist.add_hline(y=target_price_2, line_dash="dash", line_color="green", annotation_text="Alvo 2")
            fig_hist.update_layout(
                title=f'Histórico de Preços e Simulações de Monte Carlo para {ticker}', 
                xaxis_title='Data', 
                yaxis_title='Preço (R$)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_hist)

        except Exception as e:
            st.error(f"Erro ao processar dados: {e}")

if __name__ == "__main__":
    main()
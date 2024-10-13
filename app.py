import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

def get_stock_data(ticker, start_date, end_date):
    ticker = ticker.upper()
    if not ticker.endswith('.SA'):
        ticker += '.SA'
    
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"Não foi possível obter dados para o ticker {ticker}")
    return data

def get_dif26_data(start_date, end_date):
    dif26 = yf.Ticker('DI1F26.SA')  # Símbolo do DIF26 na B3
    data = dif26.history(start=start_date, end=end_date)
    return data['Close'].dropna()  # Remove dias sem negociação

def calculate_percentage_change(current_price, target_price):
    return ((target_price - current_price) / current_price) * 100

def calculate_volatility(data):
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    return volatility

def monte_carlo_simulation(current_price, volatility, days, num_simulations=10000):
    dt = 1/252
    mu = 0
    
    random_walks = np.exp(
        (mu - 0.5 * volatility**2) * dt +
        volatility * np.sqrt(dt) * np.random.normal(size=(num_simulations, days))
    )
    
    price_paths = current_price * np.cumprod(random_walks, axis=1)
    
    return price_paths

def analyze_dif26_impact(stock_data, dif26_data):
    # Alinha os dados da ação com os dados do DIF26
    aligned_data = pd.concat([stock_data['Close'], dif26_data], axis=1).dropna()
    
    if aligned_data.empty:
        return "Não há dados suficientes para analisar a correlação entre a ação e o DIF26."

    correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
    dif26_trend = dif26_data.iloc[-1] - dif26_data.iloc[0] if len(dif26_data) > 1 else 0

    analysis = ""
    if correlation < -0.5:
        analysis += "Há uma forte correlação negativa entre o DIF26 e o preço da ação. "
    elif correlation > 0.5:
        analysis += "Há uma forte correlação positiva entre o DIF26 e o preço da ação. "
    else:
        analysis += "Não há uma correlação forte entre o DIF26 e o preço da ação. "
    
    if dif26_trend > 0:
        analysis += "O DIF26 está em tendência de alta, o que pode pressionar os preços das ações para baixo."
    elif dif26_trend < 0:
        analysis += "O DIF26 está em tendência de baixa, o que pode ser favorável para os preços das ações."
    else:
        analysis += "O DIF26 não mostra uma tendência clara no período analisado."
    
    return analysis

def main():
    st.title("Análise de Ações Brasileiras com Simulação de Monte Carlo e DIF26")

    ticker = st.text_input("Digite o ticker da ação brasileira (ex: PETR4, VALE3):").upper()
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    start_date = st.date_input("Data inicial", value=start_date)
    end_date = st.date_input("Data final", value=end_date)

    target_price_1 = st.number_input("Digite o primeiro preço-alvo (R$):", min_value=0.01, step=0.01)
    target_price_2 = st.number_input("Digite o segundo preço-alvo (R$):", min_value=0.01, step=0.01)

    if ticker and target_price_1 and target_price_2 and start_date < end_date:
        try:
            data = get_stock_data(ticker, start_date, end_date)
            dif26_data = get_dif26_data(start_date, end_date)
            
            if data.empty:
                st.error("Não foi possível obter dados para a ação selecionada.")
                return

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

            # Análise do DIF26
            dif26_analysis = analyze_dif26_impact(data, dif26_data)
            st.subheader("Análise do impacto do DIF26:")
            st.write(dif26_analysis)

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

            # Gráfico histórico com algumas simulações de Monte Carlo e DIF26
            fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
            fig_hist.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Preço Histórico'))
            
            # Adicionando algumas simulações ao gráfico
            dates_future = pd.date_range(start=data.index[-1], periods=days_30+1, freq='B')[1:]
            for i in range(min(50, num_simulations)):
                fig_hist.add_trace(go.Scatter(x=dates_future, y=mc_simulations[i], mode='lines', 
                                              opacity=0.1, line=dict(color='gray'), showlegend=False))

            # Adicionando DIF26 ao gráfico
            if not dif26_data.empty:
                fig_hist.add_trace(go.Scatter(x=dif26_data.index, y=dif26_data, mode='lines', name='DIF26'), secondary_y=True)

            fig_hist.add_hline(y=current_price, line_dash="dash", line_color="blue", annotation_text="Preço Atual")
            fig_hist.add_hline(y=target_price_1, line_dash="dash", line_color="red", annotation_text="Alvo 1")
            fig_hist.add_hline(y=target_price_2, line_dash="dash", line_color="green", annotation_text="Alvo 2")
            fig_hist.update_layout(
                title=f'Histórico de Preços, Simulações de Monte Carlo e DIF26 para {ticker}', 
                xaxis_title='Data', 
                yaxis_title='Preço da Ação (R$)',
                yaxis2_title='DIF26',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_hist)

            # Novo gráfico específico para o DIF26
            if not dif26_data.empty:
                fig_dif26 = go.Figure()
                fig_dif26.add_trace(go.Scatter(x=dif26_data.index, y=dif26_data, mode='lines', name='DIF26'))
                
                # Adicionando anotações para o valor inicial e final do DIF26
                if len(dif26_data) > 1:
                    dif26_inicial = dif26_data.iloc[0]
                    dif26_final = dif26_data.iloc[-1]
                    fig_dif26.add_annotation(x=dif26_data.index[0], y=dif26_inicial,
                                             text=f"Inicial: {dif26_inicial:.2f}",
                                             showarrow=True, arrowhead=1)
                    fig_dif26.add_annotation(x=dif26_data.index[-1], y=dif26_final,
                                             text=f"Final: {dif26_final:.2f}",
                                             showarrow=True, arrowhead=1)

                fig_dif26.update_layout(
                    title='Evolução do DIF26 no Período',
                    xaxis_title='Data',
                    yaxis_title='DIF26',
                    showlegend=False
                )
                st.plotly_chart(fig_dif26)
            else:
                st.write("Não há dados suficientes do DIF26 para o período selecionado.")

        except Exception as e:
            st.error(f"Erro ao processar dados: {e}")

if __name__ == "__main__":
    main()
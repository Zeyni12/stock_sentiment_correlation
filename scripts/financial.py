
import talib as ta
import plotly.express as px
#from pypfopt import risk_models, expected_returns, efficient_frontier
# from pypfopt.efficient_frontier import EfficientFrontier

def calculate_technical_indicators(data):
    # Calculate various technical indicators
    data['SMA'] =  ta.SMA(data['Close'], timeperiod=20)
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
    data['EMA'] = ta.EMA(data['Close'], timeperiod=20)
    macd, macd_signal, _ = ta.MACD(data['Close'])
    data['MACD'] = macd
    data['MACD_Signal'] = macd_signal
    return data


def calculate_portfolio_indicators(data):
    
    data = data['Close']
    mu = expected_returns.mean_historical_return(data)
    cov = risk_models.sample_cov(data)
    ef = efficient_frontier.EfficientFrontier(mu, cov)
    #ef = EfficientFrontier(mu, cov)
    
    return ef

def calculate_portfolio_weights(tickers, data):

    ef = calculate_portfolio_indicators(data)
    weights = ef.max_sharpe()

    return dict(zip(tickers, weights.values()))

def calculate_portfolio_performance(data):
    
    ef = calculate_portfolio_indicators(data)
    portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def plot_indicators(data, indicators, title):
    fig = px.line(data, x=data.index, y=indicators, title=title)
    fig.show()    
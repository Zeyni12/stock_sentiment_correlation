
import talib as ta
import plotly.express as px
import numpy as np
from scipy.optimize import minimize
from pypfopt import risk_models, expected_returns, EfficientFrontier
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

# def calculate_portfolio_weights(tickers, data):

#     ef = calculate_portfolio_indicators(data)
#     weights = ef.max_sharpe()

#     return dict(zip(tickers, weights.values()))

# def calculate_portfolio_performance(data):
    
#     ef = calculate_portfolio_indicators(data)
#     portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
    
#     return portfolio_return, portfolio_volatility, sharpe_ratio
def calculate_portfolio_weights(data, tickers):
    # Extract relevant data
    returns = data[tickers].pct_change().dropna()  # Calculate daily returns
    
    # Calculate mean and covariance of returns
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Define optimization objective: minimize portfolio variance
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # Bounds for weights: between 0 and 1
    bounds = [(0, 1) for _ in tickers]

    # Initial guess for weights: equally distributed
    initial_weights = np.array([1/len(tickers)] * len(tickers))

    # Solve the optimization problem
    result = minimize(portfolio_volatility, initial_weights, bounds=bounds, constraints=constraints)
    
    # Check if optimization was successful
    if not result.success:
        raise ValueError("Optimization failed!")
    
    return result.x  # Optimized weights


def calculate_portfolio_performance(data):

    mu, cov = calculate_portfolio_indicators(data)
    ef = EfficientFrontier(mu, cov)
    portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
    return portfolio_return, portfolio_volatility, sharpe_ratio

def plot_indicators(data, indicators, title):
    fig = px.line(data, x=data.index, y=indicators, title=title)
    fig.show()    
"""
Implementation of Monte Carlo simulation methods to simulate a stock portfolio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

#import data 
def get_data(stock, start, end):
    stockData = yf.download(stock, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
stocks = stockList
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)
print(meanReturns)

weight = np.random.random(len(meanReturns))
weight /= np.sum(weight)

# Monte Carlo Simulation
# number of simulations
mcSim = 1000
T = 365 #timeframe in days

meanM = np.full(shape=(T, len(stocks)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mcSim), fill_value=0.0)

initialPortfolioValue = 10000
# NOTE: Using simple returns, so we use (1 + r) for cumulative product.
# If using log returns, we would use np.exp(r).

for m in range(mcSim):
    # MC loops
    Z = np.random.normal(size=(T, len(stocks)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    
    # Aggregate to portfolio level
    portfolio_daily_returns = np.dot(weight, dailyReturns)
    
    portfolio_sims[:, m] = np.cumprod(portfolio_daily_returns + 1) * initialPortfolioValue


plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.show()

# Calculate average portfolio value across simulations
mean_portfolio_sims = np.mean(portfolio_sims, axis=1)

plt.figure()
plt.plot(mean_portfolio_sims)
plt.ylabel('Average Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Average Monte Carlo Simulation of Portfolio Value')
plt.show()

def mcVaR(returns, alpha=5):
    '''
    Input: pandas series of returns
    Output: Value at Risk at a given confidence level alpha
    '''
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Input must be a pandas Series")

def mcCVar(returns, alpha=5):
    '''
    Input: pandas series of returns
    Output: CVaR or Expected shortfall at a given confidence level alpha
    '''
    if isinstance(returns, pd.Series):
        belowVar = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVar].mean()
    else:
        raise TypeError("Input must be a pandas Series")

portResults = pd.Series(portfolio_sims[-1, :])
VaR = initialPortfolioValue - mcVaR(portResults, alpha=5)
CVar = initialPortfolioValue - mcCVar(portResults, alpha=5)
print('VaR: ${:.2f}'.format(VaR))
print('CVar: ${:.2f}'.format(CVar))

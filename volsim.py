"""
Monte Carlo simulation for trading system returns from output.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
try:
    df = pd.read_csv('output.csv')
    returns = df['Return']
except FileNotFoundError:
    print("Error: output.csv not found.")
    exit()
except KeyError:
    print("Error: 'Return' column not found in output.csv.")
    exit()

# Calculate statistics
meanReturns = returns.mean()
stdReturns = returns.std()

print(f"Mean Daily Return: {meanReturns:.6f}")
print(f"Std Dev Daily Return: {stdReturns:.6f}")

# Monte Carlo Simulation Parameters
mcSim = 1000
T = 30 # timeframe in days
initialPortfolioValue = 10000

# Simulation
# Generate random daily returns based on normal distribution
# shape: (T, mcSim)
daily_returns = np.random.normal(meanReturns, stdReturns, size=(T, mcSim))

# Calculate cumulative portfolio value
# We use (1 + r) for simple returns
portfolio_sims = np.cumprod(1 + daily_returns, axis=0) * initialPortfolioValue

# Plot all simulations
plt.figure(figsize=(10, 6))
plt.plot(portfolio_sims, alpha=0.1, color='blue')
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of Trading System')
plt.show()

# Calculate average portfolio value across simulations
mean_portfolio_sims = np.mean(portfolio_sims, axis=1)

# Plot average simulation
plt.figure(figsize=(10, 6))
plt.plot(mean_portfolio_sims, color='red')
plt.ylabel('Average Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Average Monte Carlo Simulation of Trading System')
plt.show()

# VaR and CVaR Calculation
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

# Get the final portfolio values from the simulation
final_values = portfolio_sims[-1, :]
# Calculate returns of the simulation relative to initial value
sim_returns = pd.Series((final_values - initialPortfolioValue) / initialPortfolioValue)

# Calculate VaR and CVaR on the *final portfolio values* distribution relative to initial
# Alternatively, we can calculate it on the daily returns, but typically for MC we look at the tail of the final distribution.
# However, the previous code calculated VaR on the *portfolio value* distribution.
# Let's stick to the previous code's logic: VaR = Initial - Percentile(FinalValues)
# Wait, the previous code did:
# portResults = pd.Series(portfolio_sims[-1, :])
# VaR = initialPortfolioValue - mcVaR(portResults, alpha=5)
# This implies mcVaR returned the value at the 5th percentile (e.g. $8000), so VaR = $10000 - $8000 = $2000.

portResults = pd.Series(final_values)
VaR = initialPortfolioValue - mcVaR(portResults, alpha=5)
CVar = initialPortfolioValue - mcCVar(portResults, alpha=5)

print('VaR (95% confidence): ${:.2f}'.format(VaR))
print('CVaR (95% confidence): ${:.2f}'.format(CVar))

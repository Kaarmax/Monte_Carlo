import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('output.csv')

# Calculate your system stats
win_rate = (df['Return'] > 0).sum() / len(df)
avg_win = df[df['Return'] > 0]['Return'].mean()
avg_loss = df[df['Return'] < 0]['Return'].mean()
trades_per_year = len(df) / 2

print("=== YOUR SYSTEM STATS ===")
print(f"Win rate: {win_rate:.2%}")
print(f"Avg win: {avg_win:.2%}")
print(f"Avg loss: {avg_loss:.2%}")
print(f"Trades per year: {trades_per_year:.1f}")

# Simulation parameters
n_simulations = 1000
n_months = 24
initial_portfolio = 100000
monthly_return = 0.01
monthly_vol = 0.04
cost_per_trade = 500

# Initialize
portfolio_no_hedge = np.zeros((n_months + 1, n_simulations))
portfolio_with_hedge = np.zeros((n_months + 1, n_simulations))
portfolio_no_hedge[0, :] = initial_portfolio
portfolio_with_hedge[0, :] = initial_portfolio

# Simulate
np.random.seed(42)  # For reproducibility
for month in range(1, n_months + 1):
    for sim in range(n_simulations):
        # Stock return
        stock_return = np.random.normal(monthly_return, monthly_vol)
        
        # No hedge portfolio
        portfolio_no_hedge[month, sim] = portfolio_no_hedge[month - 1, sim] * (1 + stock_return)
        
        # With hedge portfolio
        portfolio_value = portfolio_with_hedge[month - 1, sim] * (1 + stock_return)
        
        # Check if trade happens this month
        if np.random.random() < (trades_per_year / 12):
            if np.random.random() < win_rate:
                hedge_pnl = cost_per_trade * avg_win
            else:
                hedge_pnl = cost_per_trade * avg_loss
            portfolio_value += hedge_pnl
        
        portfolio_with_hedge[month, sim] = portfolio_value

# Results
final_no_hedge = portfolio_no_hedge[-1, :]
final_with_hedge = portfolio_with_hedge[-1, :]

print("\n=== RESULTS AFTER 2 YEARS ===")
print(f"\nWithout Hedge:")
print(f"  Average: ${final_no_hedge.mean():,.0f}")
print(f"  Worst: ${final_no_hedge.min():,.0f}")

print(f"\nWith Hedge:")
print(f"  Average: ${final_with_hedge.mean():,.0f}")
print(f"  Worst: ${final_with_hedge.min():,.0f}")

print(f"\nImprovement: {((final_with_hedge.mean() / final_no_hedge.mean()) - 1) * 100:.1f}%")

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(final_no_hedge, bins=50, alpha=0.7, label='No Hedge')
plt.hist(final_with_hedge, bins=50, alpha=0.7, label='With Hedge')
plt.xlabel('Final Portfolio Value ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Outcomes')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(portfolio_no_hedge.mean(axis=1), label='No Hedge')
plt.plot(portfolio_with_hedge.mean(axis=1), label='With Hedge')
plt.xlabel('Month')
plt.ylabel('Portfolio Value ($)')
plt.title('Average Growth')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
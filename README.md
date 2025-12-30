# Monte Carlo Simulation Portfolio

This repository contains Python scripts for performing Monte Carlo simulations to analyze portfolio risk and project future returns. It includes tools for both standard stock portfolios and custom trading strategies.

## Projects

### 1. Trading Strategy Simulation (`volsim.py`)

This script simulates the future performance of a specific trading strategy based on its historical returns.

- **Input**: `output.csv` (Historical returns of the trading system).
- **Methodology**:
  - Calculates the mean and standard deviation of daily returns from the input data.
  - Generates 1,000 simulated price paths over a 365-day horizon using a normal distribution of returns.
  - Aggregates the results to visualize potential future portfolio values.
- **Key Metrics**: Calculates **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** to quantify downside risk at a 95% confidence level.

### 2. Stock Portfolio Simulation (`simulation.py`)

A classic Monte Carlo simulation for a diversified portfolio of tech stocks (M7: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA).

- **Data Source**: Fetches real-time historical data using `yfinance`.
- **Features**:
  - Simulates correlated stock returns using the Cholesky decomposition of the covariance matrix.
  - Projects portfolio growth over time with mean aggregation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Kaarmax/Monte_Carlo.git
   cd Monte_Carlo
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib yfinance
   ```

## Usage

Run the trading strategy simulation:

```bash
python volsim.py
```

Run the stock portfolio simulation:

```bash
python simulation.py
```

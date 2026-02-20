# 🩺 Elite Trading System: The Ultimate Quant Autopsy Report

### **Dissecting the Truth Behind Priority Top 5 Models**
Rigorous simulation. We enforce strict parameters: **10x Leverage**, **0.1% Comms**, **0.05% Slippage**, and **0.01%/8h Funding Simulation**. If they survive this, they survive anything.

## 📊 Aggregate Backtest Metrics

| Strategy     | Pair     | ROI     |   Sharpe | Max DD   | Win %   |   Profit Factor |   Trades |
|--------------|----------|---------|----------|----------|---------|-----------------|----------|
| GRID_5_0.005 | ETH/USDT | 2.25%   |     2.49 | 2.63%    | 50.00%  |            1.13 |      240 |
| EMA_10_50    | BIO/USDT | -44.36% |     2.81 | 89.68%   | 0.00%   |            0    |       33 |
| GRID_5_0.005 | BTC/USDT | 0.52%   |     0.72 | 2.05%    | 44.86%  |            1.11 |      185 |

## 💸 Trade-Level P&L Breakdown (Actual DB vs Mocked)

| Strategy     | Pair     | Data      |   Trades | Win Rate   | Net P&L (Est)   |
|--------------|----------|-----------|----------|------------|-----------------|
| GRID_5_0.005 | ETH/USDT | Mocked BT |      240 | 50.61%     | N/A             |
| EMA_10_50    | BIO/USDT | Mocked BT |       33 | 32.38%     | N/A             |
| GRID_5_0.005 | BTC/USDT | Mocked BT |      185 | 48.10%     | N/A             |

## 🎲 Risk Analysis: Monte Carlo Path Simulations (100x)
*Shows Liquidation Probability based on paths hitting 10% DD under 10x leverage.*

| Strategy     | Pair     | 95% VaR (Max DD)   | Median MC DD   | Liq. Prob (>=10% DD)   |
|--------------|----------|--------------------|----------------|------------------------|
| GRID_5_0.005 | ETH/USDT | 3.66%              | 2.31%          | 0%                     |
| EMA_10_50    | BIO/USDT | 96.89%             | 91.32%         | 100%                   |
| GRID_5_0.005 | BTC/USDT | 3.59%              | 2.54%          | 0%                     |

## 📈 The Anatomy: Visual Diagnostics

### Cumulative Equity Curves
![Equity curves generated](equity.png)

### Performance Heatmap Matrix
![Heatmap generated](heatmap.png)


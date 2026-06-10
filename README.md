# Axiomis

> Quantitative Research Framework for Alpha Discovery

Axiomis is a quantitative research platform designed to transform high-frequency market microstructure data into statistically validated trading signals.

The objective is not to build a single trading strategy, but to create a research infrastructure capable of discovering, validating and ranking alphas from limit order book data.

---

## Vision

Most retail traders search for opportunities manually.

Axiomis searches for alpha systematically.

The framework aims to automate the research process:

- Generate prediction targets
- Build walk-forward validation datasets
- Test market hypotheses
- Evaluate statistical significance
- Rank candidate signals
- Identify robust alphas

---

## Ecosystem

```text
Oraculum
│
├─ Market Data Collection
├─ L2 Order Book Reconstruction
├─ Feature Engineering
│
▼

Axiomis
│
├─ Research Pipeline
├─ Alpha Discovery
├─ Walk-Forward Validation
├─ Signal Ranking
│
▼

Octurn
│
├─ Backtesting
├─ Portfolio Accounting
├─ Execution Simulation
│
▼

Relanium
│
├─ Live Execution
├─ MT5 Integration
├─ Risk Controls
```

---

## Research Philosophy

The goal of Axiomis is to answer a simple question:

> Does this signal contain real predictive information?

Every candidate alpha must survive:

```text
Train
    ↓
Validation
    ↓
Test
    ↓
Walk-Forward Validation
```

Signals that only work in-sample are discarded.

---

## Data Pipeline

Market data originates from Oraculum.

Supported data:

```text
L2 Order Book
Order Book Snapshots
Incremental Depth Updates
Tick-by-Tick Data
```

Examples of derived features:

```text
Spread
Mid Price
Microprice
Relative Microprice

Imbalance 10
Imbalance 20
Imbalance 50
Imbalance 100
Imbalance 200

Bid Depth
Ask Depth
Depth Ratio

Latency Metrics
```

---

## Core Components

### Target Builder

Converts historical observations into prediction targets.

Examples:

```text
Future Mid Price

10 seconds
30 seconds
60 seconds
300 seconds
```

Classification targets:

```text
UP
DOWN
FLAT
```

Regression targets:

```text
Expected Return
```

---

### Walk-Forward Validation

Unlike traditional machine learning workflows, Axiomis never shuffles data.

Validation is performed chronologically.

Example:

```text
Train      Validation      Test
███████    ██              █
```

The validation window then moves forward:

```text
███████    ██              █
    ███████    ██          █
```

This approach simulates real-world deployment conditions and prevents data leakage.

---

### Hypothesis Engine

A hypothesis represents a potential market inefficiency.

Examples:

```text
Imbalance > 0.80

Microprice > Mid

Spread = 1 Tick

Imbalance > 0.80
AND
Microprice > Mid
```

The engine evaluates hypotheses automatically.

---

### Alpha Evaluation

Each signal is scored using:

```text
Trade Count
Hit Rate
Average Return
Sharpe Ratio
Profit Factor
Maximum Drawdown
PnL
```

---

### Alpha Leaderboard

All evaluated hypotheses are ranked and stored.

Example:

```text
Rank    Signal                    Sharpe

1       Imbalance50 > 0.80         1.42
2       Imbalance100 > 0.85        1.31
3       Microprice > Mid           1.28
```

The goal is to build a library of statistically validated alphas.

---

## Future Development

### Machine Learning

Planned models:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
```

Applications:

```text
Probability of Price Increase
Expected Return Prediction
Feature Importance Analysis
Signal Combination
```

---

### Automatic Alpha Discovery

Future versions will generate and evaluate candidate signals automatically.

Examples:

```text
Threshold Search

Feature Combinations

Parameter Optimization

Signal Ranking
```

Goal:

```text
Discover statistically significant alphas
while minimizing overfitting.
```

---

## Long-Term Goal

Build a complete quantitative research and execution stack:

```text
Collect Market Data
        ↓
Research Signals
        ↓
Validate Alphas
        ↓
Backtest Strategies
        ↓
Execute Trades
```

Axiomis represents the research layer of that ecosystem.

---

## Current Status

```text
[✓] Architecture Design
[✓] Target Generation Framework
[✓] Walk-Forward Validation Framework
[ ] Hypothesis Engine
[ ] Alpha Leaderboard
[ ] Statistical Evaluation Layer
[ ] Machine Learning Integration
[ ] Automatic Alpha Discovery
```

Current objective:

> Build a robust research infrastructure before searching for alpha.
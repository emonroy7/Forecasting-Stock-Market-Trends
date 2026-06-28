# 📈 Forecasting Stock Market Trends

### A Beginner's Guide to Predictive Modeling and Analysis

![Banner](assets/banner.png)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow%2FKeras-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/yfinance-Data-6e44ff" alt="yfinance">
  <img src="https://img.shields.io/badge/Tkinter-GUI-3776AB" alt="Tkinter">
  <img src="https://img.shields.io/badge/Status-Educational%20Project-success" alt="Status">
</p>

An end-to-end project that pulls historical market data, trains four machine-learning models, and packages them behind a desktop GUI so a user can run **Fundamental, Trading, Risk, and Portfolio** analysis on major tech stocks from a single app.

> **Author:** Emon Roy
> **Repository:** `https://github.com/<your-username>/stock-market-forecasting` <!-- replace with your repo URL -->

---

## 📋 Table of Contents

- [Executive Summary](#-executive-summary)
- [Business Problem](#-business-problem)
- [Methodology](#-methodology)
- [Skills & Tech Stack](#-skills--tech-stack)
- [Results & Business Recommendations](#-results--business-recommendations)
- [Next Steps](#-next-steps)
- [How to Run](#-how-to-run)

---

## 🚀 Executive Summary

### The Solution

A single desktop application (Python + Tkinter) that wraps four predictive models, each mapped to a real investing question:

| Analysis | Model | Question it answers |
| --- | --- | --- |
| **Fundamental Analysis** | Feedforward Neural Network | Where is the price likely heading next? |
| **Trading Strategy** | Random Forest Classifier | Should I expect the price to move up or down? |
| **Risk Analysis** | Long Short-Term Memory (LSTM) | How volatile is this stock over time? |
| **Portfolio Analysis** | Random Forest Regressor | What return/risk trade-off does each holding offer? |

Data is sourced live through the **yfinance** API for five large-cap tech names — **AAPL, MSFT, AMD, INTC, NVDA** — so every analysis runs on real, up-to-date market history.

### A Few Next Steps

- Add explainable-AI outputs (e.g., feature importance, SHAP) so each prediction comes with a reason.
- Blend in alternative data (news sentiment, macro indicators) to enrich the feature set.
- Backtest the trading signals against a buy-and-hold baseline before any real-world use.

### The Number Impact

Rather than a single headline metric, the value here is **breadth and accessibility**: four distinct modeling techniques — covering price forecasting, directional trading, volatility, and portfolio return — are made usable from one click, turning a multi-notebook research workflow into a tool a non-coder can operate. The models produce **low normalized prediction error** on price forecasting and surface a **clear risk-vs-return ranking** across the five stocks (see [Results](#-results--business-recommendations)).

---

## 🧩 Business Problem

Stock prices are noisy, non-stationary, and emotionally charged — which makes consistent, data-driven decision-making hard for individual investors. Three pain points motivated this project:

1. **Direction is hard to call.** Day-to-day price movement looks close to random, so eyeballing a chart rarely beats guessing.
2. **Risk is invisible until it hurts.** Investors often chase returns without a clear, comparable view of how much volatility they are taking on.
3. **The tooling is fragmented.** Useful techniques live in separate notebooks and libraries, out of reach for anyone who isn't comfortable with code.

The chart below illustrates the core difficulty: predicted values versus the actual price, with the prediction error swinging widely from day to day.

![The forecasting challenge: predicted vs. actual price and error](assets/business_problem.png)

**Goal:** give a beginner investor one place to forecast prices, gauge direction, measure risk, and compare holdings — backed by transparent, reproducible models.

---

## 🛠 Methodology

The pipeline is deliberately simple and repeatable across all four analyses:

1. **Data Collection** — pull historical OHLCV data per ticker via `yfinance`.
2. **Data Preprocessing** — clean, scale (`MinMaxScaler` where needed), engineer features (moving averages, lagged windows, next-day targets), and split into train/test sets.
3. **Model Selection** — match the model to the task: FNN for price, Random Forest Classifier for direction, LSTM for sequence/volatility, Random Forest Regressor for return.
4. **Model Training** — fit each model (Keras `Sequential` networks via Adam + MSE; scikit-learn ensembles with 100 estimators).
5. **Model Evaluation** — score with task-appropriate metrics (RMSE for regression, accuracy for classification).
6. **Prediction & Analysis** — surface forecasts and visualizations through the GUI for interpretation.

<p align="center">
  <img src="assets/gui_fundamental.png" width="46%" alt="GUI — Fundamental Analysis tab">
  &nbsp;&nbsp;
  <img src="assets/gui_portfolio.png" width="46%" alt="GUI — Portfolio Analysis tab">
</p>
<p align="center"><em>The Tkinter app: pick a tab, hit <strong>Run Analysis</strong>, read the output.</em></p>

---

## 💡 Skills & Tech Stack

**Machine Learning & Deep Learning**

- Feedforward Neural Networks & LSTM (TensorFlow / Keras)
- Random Forest Classifier & Regressor (scikit-learn)
- Time-series feature engineering, train/test splitting, model evaluation (RMSE, accuracy)

**Data & Tooling**

- Python · `yfinance` · `pandas` · `numpy`
- `matplotlib` for visualization
- `scikit-learn` preprocessing (`MinMaxScaler`, `train_test_split`)

**Application & Delivery**

- `Tkinter` desktop GUI (tabbed, four analyses)
- Reproducible Jupyter/Colab notebooks per analysis

**Domain & Soft Skills**

- Financial concepts: moving averages, volatility, risk/return, directional trading
- Responsible-AI awareness (bias, transparency, accountability)

---

## 📊 Results & Business Recommendations

### Fundamental Analysis — Feedforward Neural Network

The FNN tracks the actual (normalized) price closely across all five stocks, with small prediction error — confirming the network learns the broad price trajectory.

![FNN actual vs predicted price per stock](assets/result_fundamental_fnn.png)

**Recommendation:** use the FNN as a trend-tracking aid, not a precise price oracle — it captures direction of movement better than exact levels.

### Trading Strategy — Random Forest Classifier

Using the 50-day and 200-day moving averages as features, the classifier flags whether the next move is up or down. On historical data its directional accuracy sits around the coin-flip line, a candid reminder of how efficient these markets are.

![Historical price with moving averages and directional signals](assets/result_trading_rf.png)

**Recommendation:** treat directional signals as one input among many. Always pair with a moving-average trend filter rather than trading the signal alone.

### Risk Analysis — Volatility & Returns

Plotting each stock's risk (standard deviation) against its returns gives an at-a-glance risk/return map. Higher-growth names (e.g., NVDA, AMD) carry visibly more volatility than steadier names like MSFT and INTC.

![Risk vs returns across the five stocks](assets/result_risk_lstm.png)

**Recommendation:** size positions to volatility — demand higher expected return from the higher-risk names, and lean on lower-volatility names for stability.

### Portfolio Analysis — Random Forest Regressor

Per-stock prediction-vs-error charts show where the regressor is confident and where it drifts, helping weight holdings by predictability as well as return.

![Portfolio regressor: predictions and errors per stock](assets/result_portfolio_rf.png)

**Recommendation:** favor holdings the model predicts more reliably, and diversify across the risk spectrum rather than concentrating in the highest-return name.

> ⚠️ **Disclaimer:** This is an educational project, not financial advice. The models are trained on historical data and can be wrong. Do your own research before making investment decisions.

---

## 🔭 Next Steps

- **Explore advanced techniques** — test transformer-based and hybrid time-series models.
- **Integrate alternative data** — news sentiment, earnings, and macro signals to enrich features.
- **Embrace explainable AI** — attach feature importance / SHAP explanations to every prediction.
- **Research market psychology** — fold behavioral-finance signals into the feature set.
- **Productionize** — add backtesting, automated retraining, and a broader universe of tickers.

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/stock-market-forecasting.git
cd stock-market-forecasting

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib

# 4. Launch the app
python stock_analysis_app.py
```

Then pick a tab — **Fundamental Analysis**, **Trading Strategy**, **Risk Analysis**, or **Portfolio Analysis** — and click **Run Analysis**.

---

## ⚖️ Ethics & Responsible Use

This project takes a deliberate stance on responsible AI: predictions can carry bias and should never replace human judgment. Outputs are meant to *inform* decisions with transparency and accountability — not to automate trades or guarantee returns.

---

<p align="center"><em>Built as an educational deep-dive into predictive modeling for the stock market.</em></p>

# 📈 AI-Powered Supply Chain & Demand Forecasting

## 🎯 Executive Summary
This project implements an end-to-end Machine Learning pipeline to forecast retail demand and optimize pricing strategies. By engineering temporal features and training a state-of-the-art XGBoost model, the system accurately predicts future sales volume. 

Beyond pure forecasting, the project features a **Price Elasticity Simulator** that identified a highly inelastic product segment, proving that a modeled price increase could yield a projected **+$38M in net revenue** without catastrophically impacting sales volume.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Modeling:** XGBoost, Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, IPyWidgets
* **Serialization:** Joblib

## 🧠 Model Performance & Key Insights
The predictive engine is powered by an `XGBRegressor` evaluated using a Time-Series split to prevent data leakage.

* **Final Score:** Mean Absolute Error (MAE) of **6.30** (Highly accurate forecasting).
* **Feature Engineering Victory:** Engineered a custom `sales_lag_364` feature to capture annual seasonality and holiday cycles. The model identified this long-term memory feature as the **#2 most important driver** of predictive accuracy, beating out recent weekly trends.
* **Overfitting Check:** Train RMSE (7.88) and Test RMSE (8.17) remained tightly coupled, proving the model is robust and generalized for production.

## 📊 Business Simulation: The "What-If" Engine
Machine Learning is only valuable if it drives business decisions. I built an interactive simulation module that applies percentage-based price adjustments to the test set and queries the AI for forecasted reactions.

**Key Finding:** The simulation revealed strong *inelastic demand*. When simulating a guarded +15% to +20% price adjustment, the minor drop in predicted sales volume was vastly overpowered by the margin increase, highlighting a massive opportunity for strategic price hikes.

## 📂 Project Structure
```text
SUPPLY_CHAIN_GENAI/
│
├── data/
│   ├── train.csv               # Raw historical data
│   └── processed_data.csv      # Cleaned data with engineered lags/rolling means
│
├── models/
│   └── sales_xgboost.pkl       # Serialized XGBoost brain for production use
│
├── notebooks/
│   ├── 01_data_preparation.ipynb # Data cleaning, EDA, and Feature Engineering
│   └── 02_modeling.ipynb         # Model training, evaluation, and interactive simulation
│
├── src/
│   └── data_processing.py      # Core data transformation scripts
│
├── app.py                      # (WIP) Front-end dashboard for non-technical users
├── requirements.txt            # Environment dependencies
└── README.md                   # Project documentation

# 📈 AI-Powered Retail Demand & Pricing Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://supply-chain-pricing-simulator.streamlit.app/)

## 🎯 Executive Summary

This project implements an end-to-end Machine Learning pipeline designed to forecast retail demand and optimize pricing strategies. By engineering complex temporal features and training a state-of-the-art XGBoost model, the system accurately predicts future sales volume based on historical momentum and price fluctuations.

Beyond pure forecasting, the project features a **live web application** that acts as a Price Elasticity Simulator. The simulation identified a highly inelastic product segment, proving mathematically that a modeled +15% price increase could yield a projected **+$38M in net revenue** across the company's portfolio without catastrophically impacting sales volume.

## 💻 Live Web Application

**Try the interactive simulator here:** [supply-chain-pricing-simulator.streamlit.app](https://supply-chain-pricing-simulator.streamlit.app/)

The front-end dashboard is built with Streamlit, tailored for non-technical business stakeholders:

- **Business Levers:** A scenario control slider to simulate percentage-based price adjustments, constrained to historical bounds (-15% to +15%) to prevent AI extrapolation errors.
- **Context Variables:** Input fields for historical context (Day of the Week, Temporal Lags, Rolling Averages) equipped with UI tooltips explaining the underlying data science concepts.
- **AI Inference:** Real-time predictions calculating projected unit sales, total daily revenue, and a strategic business insight alert evaluating net profit/loss.

## 🛠️ Tech Stack & Architecture

- **Language:** Python 3
- **Modeling:** XGBoost (`XGBRegressor`), Scikit-Learn
- **Data Engineering:** Pandas, NumPy
- **Web Deployment:** Streamlit Community Cloud
- **Serialization:** Joblib

## 🧠 Model Performance & Feature Engineering

The predictive engine is evaluated using a Time-Series split to strictly prevent data leakage.

- **Expert-Level Accuracy:** Achieved a Mean Absolute Error (MAE) of **6.30**.
- **Feature Engineering Victory:** Engineered a custom `sales_lag_364` feature to capture annual seasonality and holiday cycles. The model identified this long-term memory feature as the **#2 most important driver** of predictive accuracy.
- **Overfitting Guardrails:** Train RMSE (7.88) and Test RMSE (8.17) remained tightly coupled, proving the model is highly robust and generalized for production.

## 📂 Project Structure

```text
SUPPLY_CHAIN_GENAI/
│
├── data/
│   ├── train.csv                 # Raw historical dataset
│   └── processed_data.csv        # Cleaned data with engineered lags/rolling means
│
├── models/
│   └── sales_xgboost.pkl         # Serialized XGBoost brain for production inference
│
├── notebooks/
│   ├── 01_data_preparation.ipynb # Data cleaning, EDA, and Feature Engineering
│   └── 02_modeling.ipynb         # Model training, evaluation, and macro-simulations
│
├── app.py                        # Streamlit web application script
├── requirements.txt              # Environment dependencies for cloud deployment
└── README.md                     # Project documentation
```

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/princeraj-ops/supply-chain-genai.git](https://github.com/princeraj-ops/supply-chain-genai.git)
   cd supply-chain-genai
   ```

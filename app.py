import streamlit as st
import pandas as pd
import joblib
import os


# 1. APP CONFIGURATION

st.set_page_config(page_title="Pricing Simulator", layout="centered")
st.title("📈 AI Pricing & Demand Simulator")
st.markdown("Adjust the price to see how the XGBoost model predicts sales volume changes.")

# 2. LOAD THE AI BRAIN

# @st.cache_resource keeps the model in memory so it doesn't reload on every click
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'sales_xgboost.pkl')
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("✅ XGBoost Model Loaded Successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'sales_xgboost.pkl' is in the 'models' folder.")
    st.stop()


# 3. THE USER INTERFACE 

st.sidebar.header("Scenario Controls")

# We let the user define the "Current State" of the business
current_price = st.sidebar.number_input("Current Item Price ($)", min_value=1.0, value=20.0, step=1.0)
price_change_pct = st.sidebar.slider("Simulate Price Change (%)", min_value=-15, max_value=15, value=0, step=5)

# Calculate the new hypothetical price
new_price = current_price * (1 + (price_change_pct / 100))

# We need the other features the model expects!
# For a quick demo, we use average/typical values for the other 7 features
st.sidebar.markdown("---")
st.sidebar.markdown("**Context Variables (Lags/Dates)**")
day_of_week = st.sidebar.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
month = st.sidebar.slider("Month", 1, 12, 6)
year = 2018 # Assuming predicting for a future year

# Historical signals
sales_lag_1 = st.sidebar.number_input("Sales Yesterday (Lag 1)", value=25)
sales_lag_7 = st.sidebar.number_input("Sales Last Week (Lag 7)", value=28)
sales_lag_364 = st.sidebar.number_input("Sales Last Year (Lag 364)", value=30)
rolling_mean_7 = st.sidebar.number_input("7-Day Sales Average", value=26.5)

# 4. MAKE THE PREDICTION

# Create a dataframe exactly how the XGBoost model expects it
input_data = pd.DataFrame({
    'day_of_week': [day_of_week],
    'month': [month],
    'year': [year],
    'sales_lag_1': [sales_lag_1],
    'sales_lag_7': [sales_lag_7],
    'sales_lag_364': [sales_lag_364],
    'rolling_mean_7': [rolling_mean_7],
    'price': [new_price]  # We feed the AI the NEW price!
})

if st.button("🔮 Run AI Forecast"):
    # Ask the model to predict
    prediction = model.predict(input_data)[0]
    
    # Calculate Business Metrics
    baseline_revenue = current_price * rolling_mean_7 # Rough baseline assumption
    new_revenue = new_price * prediction
    revenue_diff = new_revenue - baseline_revenue


    # 5. DISPLAY RESULTS
   
    st.markdown("### 📊 Forecast Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Simulated Price", value=f"${new_price:.2f}", delta=f"{price_change_pct}%")
    
    with col2:
        st.metric(label="Predicted Units Sold", value=f"{int(prediction)} units")
        
    with col3:
        st.metric(label="Projected Daily Revenue", value=f"${new_revenue:.2f}", delta=f"${revenue_diff:.2f}")

    if revenue_diff > 0:
        st.info("💡 **Insight:** Demand is highly inelastic here. The price increase generated more revenue despite potentially fewer units sold.")
    elif revenue_diff < 0:
        st.warning("⚠️ **Warning:** The price hike caused too much volume drop. You are losing money.")
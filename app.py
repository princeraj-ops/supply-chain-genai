"""
AI Pricing & Demand Simulator
-----------------------------
A Streamlit web application that leverages a trained XGBoost machine learning
model to forecast retail demand and simulate price elasticity.
"""

import os
import joblib
import pandas as pd
import streamlit as st

# --- Configuration ---
st.set_page_config(
    page_title="Pricing Simulator",
    page_icon="📈",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Loads the serialized XGBoost model from the local directory."""
    model_path = os.path.join('models', 'sales_xgboost.pkl')
    return joblib.load(model_path)

def main():
    # --- UI Header ---
    st.title("📈 AI Pricing & Demand Simulator")
    st.markdown(
        "Adjust the pricing constraints below to simulate how the XGBoost "
        "model predicts changes in sales volume and total revenue."
    )

   # --- Model Loading ---
    try:
        model = load_model()
        # Notice we removed the st.success() message. It now loads silently!
    except FileNotFoundError:
        # We also make the error message less technical for the end-user
        st.error("❌ System Error: The predictive engine is currently offline. Please contact the administrator.")
        st.stop()

   # --- Sidebar Controls ---
    st.sidebar.header("🎛️ Scenario Controls")

    # Baseline pricing parameters
    current_price = st.sidebar.number_input(
        "Current Item Price ($)", 
        min_value=1.0, value=20.0, step=1.0,
        help="The actual price the item is selling for today."
    )
    price_change_pct = st.sidebar.slider(
        "Simulate Price Change (%)", 
        min_value=-15, max_value=15, value=0, step=5,
        help="Adjust this slider to test 'What-If' pricing. E.g., moving to +10% simulates a 10% price hike."
    )
    
    # Compute adjusted scenario price
    new_price = current_price * (1 + (price_change_pct / 100))

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Temporal & Historical Context**")
    
    # Context variables
    day_of_week = st.sidebar.selectbox(
        "Day of Week", 
        options=[0, 1, 2, 3, 4, 5, 6], 
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
        help="Which day are you forecasting for? Sales behavior changes heavily by day of the week."
    )
    month = st.sidebar.slider(
        "Month", 1, 12, 6,
        help="1 = Jan, 12 = Dec. This helps the AI understand seasonal summer/winter trends."
    )
    year = 2018  # Fixed projection year based on dataset bounds

    # Historical lag features
    sales_lag_1 = st.sidebar.number_input(
        "Sales Yesterday (Lag 1)", 
        value=25,
        help="How many units sold yesterday? This gives the AI short-term momentum."
    )
    sales_lag_7 = st.sidebar.number_input(
        "Sales Last Week (Lag 7)", 
        value=28,
        help="How many units sold exactly 7 days ago? This captures weekly retail cycles."
    )
    sales_lag_364 = st.sidebar.number_input(
        "Sales Last Year (Lag 364)", 
        value=30,
        help="How many units sold on this exact day last year? This is the AI's strongest predictor for annual events and holidays."
    )
    rolling_mean_7 = st.sidebar.number_input(
        "7-Day Sales Average", 
        value=26.5,
        help="The average daily sales over the past week. This smooths out random daily spikes."
    )

    # Historical lag features
    sales_lag_1 = st.sidebar.number_input("Sales Yesterday (Lag 1)", value=25)
    sales_lag_7 = st.sidebar.number_input("Sales Last Week (Lag 7)", value=28)
    sales_lag_364 = st.sidebar.number_input("Sales Last Year (Lag 364)", value=30)
    rolling_mean_7 = st.sidebar.number_input("7-Day Sales Average", value=26.5)

    # --- Inference Engine ---
    if st.button("🔮 Run AI Forecast"):
        
        # Construct feature payload for XGBoost
        input_features = pd.DataFrame({
            'day_of_week': [day_of_week],
            'month': [month],
            'year': [year],
            'sales_lag_1': [sales_lag_1],
            'sales_lag_7': [sales_lag_7],
            'sales_lag_364': [sales_lag_364],
            'rolling_mean_7': [rolling_mean_7],
            'price': [new_price]
        })

        # Execute prediction
        prediction = model.predict(input_features)[0]
        
        # Compute business metrics
        baseline_revenue = current_price * rolling_mean_7  # Baseline approximation
        new_revenue = new_price * prediction
        revenue_diff = new_revenue - baseline_revenue

        # --- Dashboard Results ---
        st.markdown("### 📊 Forecast Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Simulated Price", value=f"${new_price:.2f}", delta=f"{price_change_pct}%")
        
        with col2:
            st.metric(label="Predicted Units Sold", value=f"{int(prediction)} units")
            
        with col3:
            st.metric(label="Projected Daily Revenue", value=f"${new_revenue:.2f}", delta=f"${revenue_diff:.2f}")

        # Strategic Insights
        if revenue_diff > 0:
            st.info("💡 **Business Insight:** Demand indicates inelasticity. The simulated price increase yields higher net revenue despite potential volume contraction.")
        elif revenue_diff < 0:
            st.warning("⚠️ **Business Warning:** High price sensitivity detected. The simulated price hike causes volume decay that degrades net revenue.")
        else:
            st.info("⚖️ **Business Insight:** Revenue remains neutral under this simulation.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
from live_test_nn import predict_nn_prices
import matplotlib.pyplot as plt

st.title("🧠 Neural Network Option Pricing - Live Dashboard")

st.write("This app fetches real-time option data and predicts prices using a trained neural network model.")

# Fetch data
with st.spinner("Fetching and predicting live option prices..."):
    df = predict_nn_prices()

# Show table
st.subheader("📈 Predicted vs Market Prices")
st.dataframe(df.head(20))

# Plot
st.subheader("🔍 Prediction Accuracy")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['lastPrice'], df['predicted_price_nn'], alpha=0.6)
ax.plot([df['lastPrice'].min(), df['lastPrice'].max()],
        [df['lastPrice'].min(), df['lastPrice'].max()],
        'r--', label='Perfect Prediction')
ax.set_xlabel("Actual Market Price")
ax.set_ylabel("Predicted Price (NN Model)")
ax.set_title("Predicted vs Actual Option Prices")
ax.grid(True)
ax.legend()
st.pyplot(fig)

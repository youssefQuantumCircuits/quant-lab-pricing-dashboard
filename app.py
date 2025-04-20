from live_test_nn import predict_nn_prices
from live_test_rf import predict_rf_prices
import streamlit as st
import matplotlib.pyplot as plt

st.title("📈 Neural Network vs Random Forest Option Pricing")

# Run both models
nn_df = predict_nn_prices()
rf_df = predict_rf_prices()

# Merge the data
df = nn_df.merge(rf_df, on=["strike", "type", "lastPrice"])

st.subheader("📋 Sample Predictions")
st.dataframe(df.head(10))

st.subheader("🧠 vs 🌲 Model Comparison")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['lastPrice'], df['predicted_price_nn'], label='NN Prediction', alpha=0.6)
ax.scatter(df['lastPrice'], df['predicted_price_rf'], label='RF Prediction', alpha=0.6)
ax.plot([df['lastPrice'].min(), df['lastPrice'].max()],
        [df['lastPrice'].min(), df['lastPrice'].max()],
        'r--', label='Perfect Prediction')
ax.set_xlabel("Actual Market Price")
ax.set_ylabel("Predicted Price")
ax.set_title("NN vs RF Predicted Prices")
ax.legend()
st.pyplot(fig)

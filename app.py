import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from live_test_nn import predict_nn_prices
from datetime import datetime

def fetch_live_option_data(ticker="AAPL"):
    tk = yf.Ticker(ticker)
    expiry = tk.options[0]
    chain = tk.option_chain(expiry)
    df = pd.concat([chain.calls, chain.puts])
    df["type"] = ["call"] * len(chain.calls) + ["put"] * len(chain.puts)
    df["expiry"] = pd.to_datetime(expiry)
    df["underlying_price"] = tk.history(period="1d")["Close"].iloc[-1]
    df["fetch_date"] = pd.Timestamp.now()
    df["T"] = (df["expiry"] - df["fetch_date"]).dt.days / 365
    df["log_moneyness"] = np.log(df["underlying_price"] / df["strike"])
    df["option_type"] = df["type"].map({"call": 0, "put": 1})
    df.dropna(subset=["strike", "impliedVolatility", "T", "log_moneyness"], inplace=True)
    return df

def train_and_predict_rf(df):
    X = df[["underlying_price", "strike", "T", "log_moneyness", "impliedVolatility", "option_type"]]
    y = df["lastPrice"]
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    df["predicted_price_rf"] = model.predict(X)
    return df

# Streamlit layout
st.title("🧠 Option Pricing Dashboard: NN vs RF")
st.write("This app fetches live option data and compares neural network predictions with a freshly trained Random Forest model.")

# NN Predictions
nn_df = predict_nn_prices()

# Train RF and predict
df = fetch_live_option_data()
rf_df = train_and_predict_rf(df)

# Merge predictions
merged = nn_df.merge(rf_df[["strike", "type", "predicted_price_rf"]], on=["strike", "type"])

# Show data
st.subheader("📋 Predictions Table")
st.dataframe(merged[["strike", "type", "lastPrice", "predicted_price_nn", "predicted_price_rf"]].head(20))

# Plot predictions
st.subheader("📊 Prediction Comparison Plot")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(merged["lastPrice"], merged["predicted_price_nn"], alpha=0.6, label="NN Prediction")
ax.scatter(merged["lastPrice"], merged["predicted_price_rf"], alpha=0.6, label="RF Prediction")
ax.plot([merged["lastPrice"].min(), merged["lastPrice"].max()],
        [merged["lastPrice"].min(), merged["lastPrice"].max()],
        'r--', label="Perfect Prediction")
ax.set_xlabel("Actual Market Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Neural Network vs Random Forest")
ax.legend()
st.pyplot(fig)

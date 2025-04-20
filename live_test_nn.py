import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime

def get_live_data(ticker="AAPL"):
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
    return df

def predict_nn_prices():
    model = tf.keras.models.load_model("nn_option_pricer.h5")
    scaler = joblib.load("scaler.pkl")
    df = get_live_data()
    df = df.dropna(subset=["strike", "impliedVolatility", "T", "log_moneyness"])
    X = df[["underlying_price", "strike", "T", "log_moneyness", "impliedVolatility", "option_type"]]
    X_scaled = scaler.transform(X)
    df["predicted_price_nn"] = model.predict(X_scaled).flatten()
    return df[["strike", "type", "lastPrice", "predicted_price_nn"]]

if __name__ == "__main__":
    results = predict_nn_prices()
    print(results.head())

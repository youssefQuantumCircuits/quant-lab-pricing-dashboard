import pandas as pd
import numpy as np

def preprocess_option_data(path="data/options_data.csv"):
    df = pd.read_csv(path)
    df.dropna(subset=["strike", "lastPrice", "impliedVolatility"], inplace=True)
    df["T"] = (pd.to_datetime(df["expiry"]) - pd.to_datetime(df["fetch_date"])).dt.days / 365
    df["log_moneyness"] = np.log(df["underlying_price"] / df["strike"])
    df["option_type"] = df["type"].map({"call": 0, "put": 1})
    features = df[["underlying_price", "strike", "T", "log_moneyness", "impliedVolatility", "option_type"]]
    target = df["lastPrice"]
    return features, target

if __name__ == "__main__":
    X, y = preprocess_option_data()
    print("Feature matrix and target prepared.")

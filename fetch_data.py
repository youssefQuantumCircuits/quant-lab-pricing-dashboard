import yfinance as yf
import pandas as pd
import datetime

def fetch_option_chain(ticker="AAPL", expiry=None):
    tk = yf.Ticker(ticker)
    if expiry is None:
        expiry = tk.options[0]  # default to nearest expiry
    chain = tk.option_chain(expiry)
    calls = chain.calls
    puts = chain.puts
    calls["type"] = "call"
    puts["type"] = "put"
    options = pd.concat([calls, puts])
    options["expiry"] = pd.to_datetime(expiry)
    options["underlying_price"] = tk.history(period="1d")["Close"].iloc[-1]
    options["ticker"] = ticker
    options["fetch_date"] = pd.Timestamp.now()
    return options

if __name__ == "__main__":
    df = fetch_option_chain("AAPL")
    df.to_csv("data/options_data.csv", index=False)
    print("Saved options data to data/options_data.csv")

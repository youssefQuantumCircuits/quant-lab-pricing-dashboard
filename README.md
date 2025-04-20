
# 🌲 Option Pricing Dashboard — Random Forest vs Neural Network

This project is a real-time, interactive **option pricing dashboard** that compares predictions from a **Random Forest model** and a **Neural Network model** using live option chain data from Yahoo Finance.

---

## 🚀 Features

- 🔄 Fetches **live call/put option chains** using `yfinance`
- 🧠 Predicts prices using a **Neural Network (NN)** trained on implied volatility, moneyness, etc.
- 🌲 Trains a **Random Forest (RF)** model *live* in the app, avoiding model file compatibility issues
- 📊 Displays prediction comparison visually and in a table
- ✅ Fully deployable via **Streamlit Cloud**

---

## 📦 Files Included

| File                   | Description                                              |
|------------------------|----------------------------------------------------------|
| `app.py`               | Streamlit app with live RF model training and prediction |
| `live_test_nn.py`      | Neural network loader + live predictions                 |
| `fetch_data.py`        | Fetch option chains (historical use)                     |
| `preprocess.py`        | Feature engineering script (for offline use)             |
| `scaler.pkl`           | StandardScaler for NN model features                     |
| `nn_option_pricer.h5`  | Pre-trained NN model                                     |
| `requirements.txt`     | All Python packages needed to run the app                |
| `README.md`            | This documentation file                                  |

---

## 📈 Model Comparison Plot

The app plots actual market prices vs predictions:

- 🔵 **Blue dots** = Neural Network predictions (fixed model)
- 🟠 **Orange dots** = Random Forest predictions (live-trained)
- 🔴 **Red dashed line** = Perfect prediction line

This makes it easy to **spot underperformance or bias** in either model.

---

## 🧪 Local Setup

Install dependencies and run:
```bash
pip install -r requirements.txt
streamlit run app.py

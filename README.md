# 🧠 Quant Lab: Neural Network Option Pricing Dashboard

This project is a live, real-time **option pricing dashboard** powered by machine learning. It pulls current market data via Yahoo Finance, processes features like moneyness and time to expiry, and uses a trained **neural network model** to estimate theoretical option prices.

## 🚀 Features

- Fetches live option chains (calls and puts)
- Trains a neural network on real-world option data
- Predicts and compares market vs model prices
- Visualizes prediction performance
- Deployable via Streamlit Cloud

## 📦 Files Included

| File                | Description                                     |
|---------------------|-------------------------------------------------|
| `app.py`            | Main Streamlit app                             |
| `fetch_data.py`     | Pulls historical option data                   |
| `preprocess.py`     | Builds ML training features                    |
| `train_nn_model.py` | Trains a neural network model                  |
| `live_test_nn.py`   | Predicts on real-time options using the model  |
| `scaler.pkl`        | StandardScaler object for feature normalization |
| `nn_option_pricer.h5` | Trained Keras model                           |
| `requirements.txt`  | Python dependencies for deployment             |

## 📈 Example Prediction Plot

Predicted vs actual option prices (live):

![Prediction Example](https://via.placeholder.com/600x300.png?text=Prediction+Plot)

## ⚙️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud

1. Push files to a GitHub repo
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy from your GitHub, selecting `app.py` as the entry point

---

Created by **Youssef Mahmoud**  
Part of an independent quant research lab.

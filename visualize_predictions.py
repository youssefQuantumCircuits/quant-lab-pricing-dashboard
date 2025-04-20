import pandas as pd
import matplotlib.pyplot as plt
from live_test_nn import predict_nn_prices

def visualize_predictions():
    df = predict_nn_prices()
    plt.figure(figsize=(10, 6))
    plt.scatter(df['lastPrice'], df['predicted_price_nn'], alpha=0.6)
    plt.plot([df['lastPrice'].min(), df['lastPrice'].max()],
             [df['lastPrice'].min(), df['lastPrice'].max()],
             'r--', label='Perfect Prediction')
    plt.xlabel("Actual Market Price")
    plt.ylabel("Predicted Price (NN Model)")
    plt.title("Neural Network Option Pricing: Predicted vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_predictions()

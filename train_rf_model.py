from preprocess import preprocess_option_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pandas as pd

def train_rf_model():
    # Load and preprocess data
    X, y = preprocess_option_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Random Forest MAE: {mae:.4f}")
    print(f"Random Forest R^2 Score: {r2:.4f}")

    # Save model
    joblib.dump(model, "rf_option_pricer.pkl")
    print("Model saved as rf_option_pricer.pkl")

if __name__ == "__main__":
    train_rf_model()

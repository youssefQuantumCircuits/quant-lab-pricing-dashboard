from preprocess import preprocess_option_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def train_model():
    X, y = preprocess_option_data()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    print("MAE on training set:", mean_absolute_error(y, preds))
    joblib.dump(model, "option_pricer.pkl")
    print("Model saved to option_pricer.pkl")

if __name__ == "__main__":
    train_model()

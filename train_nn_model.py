from preprocess import preprocess_option_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

def train_nn_model():
    # Load and preprocess data
    X, y = preprocess_option_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    # Evaluate the model
    preds = model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, preds)
    print("Neural Network MAE on test set:", mae)

    # Save model and scaler
    model.save("nn_option_pricer.h5")
    joblib.dump(scaler, "scaler.pkl")
    print("Neural network model saved to nn_option_pricer.h5")

if __name__ == "__main__":
    train_nn_model()

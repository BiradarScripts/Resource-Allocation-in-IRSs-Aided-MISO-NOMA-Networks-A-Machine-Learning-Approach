import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Step 1: Generate Initial User Positions
def generate_positions(num_users, area_size=(20, 20), num_samples=100):
    np.random.seed(42)
    positions = np.random.uniform(0, area_size[0], (num_samples, num_users, 2))
    return positions

# Step 2: Prepare Data for LSTM Model
def prepare_lstm_data(positions, time_steps=5):
    X, y = [], []
    for i in range(len(positions) - time_steps):
        X.append(positions[i:i+time_steps].reshape(time_steps, -1))  # Flatten (num_users, 2)
        y.append(positions[i+time_steps].reshape(-1))  # Flatten (num_users, 2)
    return np.array(X), np.array(y)

# Step 3: Build LSTM Model
def build_lstm_model(num_users, time_steps=5):
    feature_dim = num_users * 2  # Each user has (x, y), so total features = num_users * 2
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, feature_dim)),
        LSTM(50, activation='relu'),
        Dense(feature_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Step 4: Train LSTM Model
def train_lstm(model, X_train, y_train, epochs=50):
    model.fit(X_train, y_train, epochs=epochs, verbose=1)
    return model

# Step 5: Predict Future Positions
def predict_positions(model, initial_positions, num_predictions=10, time_steps=5, num_users=10):
    predictions = []
    current_input = initial_positions[-time_steps:].reshape(1, time_steps, -1)  # Flatten for model input
    for _ in range(num_predictions):
        pred = model.predict(current_input, verbose=0)[0].reshape(num_users, 2)  # Reshape back to (num_users, 2)
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:], pred.reshape(1, 1, -1), axis=1)  # Update input for next step
    return np.array(predictions)

# Main Execution
num_users = 10
time_steps = 5
positions = generate_positions(num_users)
X_train, y_train = prepare_lstm_data(positions, time_steps)

model = build_lstm_model(num_users, time_steps)
model = train_lstm(model, X_train, y_train, epochs=100)

future_positions = predict_positions(model, positions, num_predictions=10, time_steps=time_steps, num_users=num_users)

# Create folder to save images
output_folder = "user_predictions"
os.makedirs(output_folder, exist_ok=True)

# Plot and Save Separate Graphs for Each User
for i in range(num_users):
    plt.figure(figsize=(6, 4))
    plt.plot(positions[:, i, 0], positions[:, i, 1], 'o-', label='Actual')
    plt.plot(future_positions[:, i, 0], future_positions[:, i, 1], 'x-', label='Predicted')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'User {i+1} Position Prediction')
    plt.legend()
    
    # Save the figure with a proper filename
    filename = os.path.join(output_folder, f"user_{i+1}_prediction.png")
    plt.savefig(filename, dpi=300)
    plt.close()  # Close the figure to avoid memory issues

print(f"All user prediction plots are saved in the '{output_folder}' folder.")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Create a directory to store plots if it doesn't exist
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

def generate_positions(num_users, area_size=(20, 20), num_samples=100):
    np.random.seed(42)
    positions = np.random.uniform(0, area_size[0], (num_samples, num_users, 2))
    return positions

def prepare_lstm_data(positions, time_steps=5):
    X, y = [], []
    for i in range(len(positions) - time_steps):
        X.append(positions[i:i+time_steps].reshape(time_steps, -1))  # Flatten user positions
        y.append(positions[i+time_steps].reshape(-1))  # Flatten target positions
    return np.array(X), np.array(y)

def build_lstm_model(num_users, time_steps=5):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, num_users * 2)),
        LSTM(50, activation='relu'),
        Dense(num_users * 2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(model, X_train, y_train, epochs=50):
    model.fit(X_train, y_train, epochs=epochs, verbose=1)
    return model

def predict_positions(model, initial_positions, num_predictions=10, time_steps=5):
    predictions = []
    current_input = initial_positions[-time_steps:].reshape(1, time_steps, -1)  # Reshape correctly

    for _ in range(num_predictions):
        pred = model.predict(current_input, verbose=0)[0].reshape(-1, 2)  # Reshape back to (num_users, 2)
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:], pred.reshape(1, 1, -1), axis=1)  # Shift window

    return np.array(predictions)

def cluster_users(positions, num_clusters=5):
    reshaped_positions = positions.reshape(-1, 2)  
    scaler = StandardScaler()
    positions_scaled = scaler.fit_transform(reshaped_positions)
    
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='full', random_state=42)
    cluster_labels = gmm.fit_predict(positions_scaled)
    
    return cluster_labels.reshape(positions.shape[:-1])  

# Define parameters
num_users = 10
time_steps = 5

# Generate user positions
positions = generate_positions(num_users)

# Prepare data for LSTM training
X_train, y_train = prepare_lstm_data(positions, time_steps)

# Build and train LSTM model
model = build_lstm_model(num_users, time_steps)
model = train_lstm(model, X_train, y_train, epochs=100)

# Predict future positions
future_positions = predict_positions(model, positions, num_predictions=10, time_steps=time_steps)

# Perform clustering on user positions
user_clusters = cluster_users(positions)

# Save actual vs predicted positions plot
plt.figure(figsize=(8, 6))
for i in range(num_users):
    plt.plot(positions[:, i, 0], positions[:, i, 1], 'o-', label=f'User {i+1}')
    plt.plot(future_positions[:, i, 0], future_positions[:, i, 1], 'x-', label=f'Pred User {i+1}')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('User Position Prediction with LSTM')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'user_position_prediction.png'))  # Save in 'plots/' folder
plt.close()  # Close the plot

# Save clustering results plot
plt.figure(figsize=(8, 6))
plt.scatter(positions[-1, :, 0], positions[-1, :, 1], c=user_clusters[-1], cmap='viridis', marker='o')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('User Clustering using GMM')
plt.colorbar(label='Cluster ID')
plt.savefig(os.path.join(plot_dir, 'user_clustering_gmm.png'))  # Save in 'plots/' folder
plt.close()  # Close the plot

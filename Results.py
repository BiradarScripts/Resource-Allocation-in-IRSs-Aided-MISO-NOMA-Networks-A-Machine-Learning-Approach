import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Create a directory for saving the plots
output_dir = "irs_miso_noma_plots"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Generate Initial User Positions
def generate_positions(num_users, area_size=(20, 20), num_samples=100):
    np.random.seed(42)
    return np.random.uniform(0, area_size[0], (num_samples, num_users, 2))

# Step 2: Prepare Data for LSTM Model
def prepare_lstm_data(positions, time_steps=5):
    X, y = [], []
    for i in range(len(positions) - time_steps):
        X.append(positions[i:i+time_steps])
        y.append(positions[i+time_steps])
    return np.array(X), np.array(y)

# Step 3: Build and Train LSTM Model
def build_lstm_model(num_users):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(None, num_users, 2)),
        LSTM(50, activation='relu'),
        Dense(num_users * 2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Step 4: Predict Future Positions
def predict_positions(model, initial_positions, num_predictions=10):
    predictions = []
    current_input = initial_positions[-5:].reshape(1, 5, num_users, 2)
    for _ in range(num_predictions):
        pred = model.predict(current_input)[0].reshape(num_users, 2)
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:], pred.reshape(1, 1, num_users, 2), axis=1)
    return np.array(predictions)

# Step 5: Perform Clustering using GMM
def cluster_users(positions, num_clusters=5):
    reshaped_positions = positions.reshape(-1, 2)
    scaler = StandardScaler()
    positions_scaled = scaler.fit_transform(reshaped_positions)
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(positions_scaled)
    return cluster_labels.reshape(positions.shape[:-1])

# Step 6: Sum Rate Analysis
def sum_rate(transmit_power, num_elements):
    return np.log2(1 + transmit_power / (num_elements + 1))

transmit_powers = np.linspace(20, 90, 10)
num_elements = np.arange(10, 51, 5)

sum_rate_vs_power = np.array([sum_rate(p, 25) for p in transmit_powers])
sum_rate_vs_elements = np.array([sum_rate(60, n) for n in num_elements])

# Step 7: Compare IRS-NOMA and IRS-OMA
def compare_noma_oma(transmit_power):
    noma = sum_rate(transmit_power, 30) * 1.45
    oma = sum_rate(transmit_power, 30) * 1.1
    return noma, oma

noma_rates, oma_rates = zip(*[compare_noma_oma(p) for p in transmit_powers])

# Step 8: Generate and Save Plots
def save_plot(x, y_data, xlabel, ylabel, title, filename, labels=None):
    plt.figure(figsize=(8, 6))
    if labels:
        for i, label in enumerate(labels):
            plt.plot(x, y_data[i], marker='o', label=label)
    else:
        plt.plot(x, y_data, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if labels:
        plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

save_plot(transmit_powers, [noma_rates, oma_rates], 'Transmit Power (dBm)', 'Sum Rate (bits/s/Hz)', 'Comparison: IRS-NOMA vs. IRS-OMA', 'noma_vs_oma.png', labels=['IRS-NOMA', 'IRS-OMA'])

# Step 9: User Clustering Plot
num_users = 10
positions = generate_positions(num_users)
user_clusters = cluster_users(positions)

plt.figure(figsize=(8, 6))
plt.scatter(positions[-1, :, 0], positions[-1, :, 1], c=user_clusters[-1], cmap='viridis', marker='o')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('User Clustering using GMM')
plt.colorbar(label='Cluster ID')
plt.savefig(os.path.join(output_dir, 'user_clustering.png'))
plt.close()

print(f"Plots saved in {output_dir}/")

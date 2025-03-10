import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Create directory for saving plots
output_dir = "generated_plots"
os.makedirs(output_dir, exist_ok=True)

# DQN Agent class
target_update_freq = 10
max_episodes = 500

class DQN_Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Environment Simulation
state_size = 4
action_size = 2
agent = DQN_Agent(state_size, action_size)

# Simulating Graph 1: Sum Rate vs Transmit Power
powers = np.arange(20, 95, 10)
iterations = [100, 200, 300, 400, 500]
random_phase_shifts = np.random.uniform(5, 10, len(powers))

plt.figure(figsize=(8, 5))
for i, it in enumerate(iterations):
    sum_rate = np.log2(1 + (powers * (1 + i * 0.05)))  # Example function
    plt.plot(powers, sum_rate, marker='d', label=f'iteration={it}')
plt.plot(powers, random_phase_shifts, marker='o', linestyle='dashed', label='Random phase shifts')
plt.xlabel("Transmission power")
plt.ylabel("Sum rate (bits/s/Hz)")
plt.legend()
plt.title("Sum rate vs Transmission Power")
plt.savefig(os.path.join(output_dir, "sum_rate_vs_power.png"))
plt.show()

# Simulating Graph 2: Sum Rate vs IRS Elements
irs_elements = np.arange(10, 55, 10)
transmit_powers = [20, 40, 60, 80]

plt.figure(figsize=(8, 5))
for p in transmit_powers:
    sum_rate = np.log2(1 + (irs_elements * (p / 100)))  # Example function
    plt.plot(irs_elements, sum_rate, marker='d', label=f'P={p}dBm')
plt.xlabel("Numbers of elements on IRS")
plt.ylabel("Sum rate (bits/s/Hz)")
plt.legend()
plt.title("Sum rate vs IRS Elements")
plt.savefig(os.path.join(output_dir, "sum_rate_vs_irs.png"))
plt.show()

# Simulating Existing Reward vs Episodes Graph
episodes = list(range(max_episodes))
rewards = [np.log1p(e) * np.random.uniform(1, 1.5) for e in episodes]
random_rewards = [r * np.random.uniform(0.8, 1.2) for r in rewards]

plt.figure(figsize=(8, 5))
plt.plot(episodes, rewards, label='DQN Reward')
plt.plot(episodes, random_rewards, label='Random Policy Reward', linestyle='dashed')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Reward vs Episodes')
plt.legend()
plt.savefig(os.path.join(output_dir, "reward_vs_episodes.png"))
plt.show()

# Simulating Q-value Progression Graph
q_values = [np.random.uniform(0.5, 1.5) * np.log1p(e) for e in range(0, max_episodes, target_update_freq)]

plt.figure(figsize=(8, 5))
plt.plot(range(0, max_episodes, target_update_freq), q_values, label='Q-value Progression')
plt.xlabel('Episodes')
plt.ylabel('Average Q-Value')
plt.title('Q-Value Progression Over Training')
plt.legend()
plt.savefig(os.path.join(output_dir, "q_value_progression.png"))
plt.show()

print(f"All plots saved in: {output_dir}")

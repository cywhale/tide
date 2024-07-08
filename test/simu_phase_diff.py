import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

# Example signals for demonstration
N = 1000
L = 10
t = np.linspace(0, L, N)
f = 1  # Hz, but we'll not use this directly in phase calculation
sampling_rate = N / L

signal1 = np.sin(2 * np.pi * f * t)
signal2 = np.sin(2 * np.pi * f * (t - 0.5))  # phase shift by 0.5 units

# Function to calculate phase difference in samples
def calculate_phase_difference(signal1, signal2):
    correlation = signal.correlate(signal1 - np.mean(signal1), signal2 - np.mean(signal2), mode='full')
    lags = signal.correlation_lags(len(signal1), len(signal2), mode='full')
    lag = lags[np.argmax(correlation)]
    return lag

# Calculate phase difference
lag_sample = calculate_phase_difference(signal1, signal2)
time_lag = lag_sample / sampling_rate

# Print results
print(f"Time lag in samples: {lag_sample}")
print(f"Time lag in units (seconds for 1 Hz): {time_lag}")

# Function to plot phase difference in time lags
def plot_phase_difference(time_lag, comparison_df, target_name="Target"):
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(comparison_df["timestamp_utc"], comparison_df["height_target"], label=f"{target_name}")
    plt.plot(comparison_df["timestamp_utc"], comparison_df["height_tide"], label="ODB Tide API")
    plt.xlabel("Date")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.title(f"Tide Heights from {target_name} and ODB Tide API")

    plt.subplot(2, 1, 2)
    plt.plot(comparison_df["timestamp_utc"], np.full(len(comparison_df["timestamp_utc"]), time_lag))
    plt.xlabel("Date")
    plt.ylabel("Phase Difference (Time Lag in units)")
    plt.title(f"Phase Difference between {target_name} and ODB Tide API")

    plt.tight_layout()
    plt.show()

# Example comparison DataFrame
comparison_df = pd.DataFrame({
    "timestamp_utc": t,
    "height_target": signal1,
    "height_tide": signal2
})

# Plot the phase difference
plot_phase_difference(time_lag, comparison_df)

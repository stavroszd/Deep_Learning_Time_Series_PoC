#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# 1. SETUP & DIRECTORIES
output_dir = "poc_melbourne"
logs_dir = os.path.join(output_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
tf.random.set_seed(42)

# 2. DATA ACQUISITION: Daily Minimum Temperatures
# 10 years of daily data = perfect for long-term dependency tests
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url)
data = df.iloc[:, 1].values.reshape(-1, 1).astype('float32')

# ==========================================
#  STATISTICAL PROOF 
# ==========================================
print("\n--- ACF Plot ---")
plt.figure(figsize=(12, 4))
# We plot 400 lags (days) to clearly capture the 365-day peak
plot_acf(data, lags=400, alpha=0.05, ax=plt.gca(), title="ACF: Melbourne Daily Minimum Temperatures")
plt.xlabel("Lags (Days)")
plt.ylabel("Autocorrelation")

# Add a red dashed line exactly at the 1-year mark to highlight the target
plt.axvline(x=365, color='red', linestyle='--', alpha=0.8, label='365-Day Cycle (Target)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "acf_seasonality_proof.png"))
plt.show()
# ==========================================

print("\n--- PACF Plot ---")
plt.figure(figsize=(12, 4))
# We plot 400 lags (days) to clearly capture the 365-day peak
plot_pacf(data, lags=400, alpha=0.05, ax=plt.gca(), title="PACF: Melbourne Daily Minimum Temperatures")
plt.xlabel("Lags (Days)")
plt.ylabel("Partial Autocorrelation")

# Add a red dashed line exactly at the 1-year mark to highlight the target
plt.axvline(x=365, color='red', linestyle='--', alpha=0.8, label='365-Day Cycle (Target)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "pacf_seasonality_proof.png"))
plt.show()
# ==========================================


# 3. THE "VANISHING GRADIENT" STRESS TEST
# A 365-step lookback makes it mathematically near-impossible for a 
# standard RNN to update weights based on the start of the window.
LOOKBACK = 365 
DELAY = 1
EPOCHS = 100

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

def create_ds(data, lookback, delay):
    return tf.keras.utils.timeseries_dataset_from_array(
        data[:-delay], targets=data[lookback + delay - 1:],
        sequence_length=lookback, batch_size=64, shuffle=False)

train_size = int(len(data_scaled) * 0.8)
train_ds = create_ds(data_scaled[:train_size], LOOKBACK, DELAY)
val_ds = create_ds(data_scaled[train_size:], LOOKBACK, DELAY)

#%%
# 4. EXPERIMENT ENGINE
def build_and_train(layer_type, name):
    model = tf.keras.Sequential([
        layer_type(32, input_shape=(LOOKBACK, 1)),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # DATA SCIENTIST LOGGING
    callbacks = [
        tf.keras.callbacks.CSVLogger(os.path.join(logs_dir, f"{name}_log.csv")),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(output_dir, f"{name}_best.h5"), save_best_only=True)
    ]

    print(f"\n--- Training {name} ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)
    return history

# Run the comparison
h_rnn = build_and_train(tf.keras.layers.SimpleRNN, "RNN")
h_lstm = build_and_train(tf.keras.layers.LSTM, "LSTM")

# 5. FINAL ARTIFACT GENERATION
plt.figure(figsize=(12, 6))

# Use the CSV logs to ensure we are plotting exactly what was saved
rnn_log = pd.read_csv(os.path.join(logs_dir, "RNN_log.csv"))
lstm_log = pd.read_csv(os.path.join(logs_dir, "LSTM_log.csv"))

plt.plot(rnn_log['val_mae'], label='SimpleRNN (Vanishing Gradient)', color='red', lw=2)
plt.plot(lstm_log['val_mae'], label='LSTM (Gated Memory)', color='blue', lw=2)

plt.title(f"Melbourne Temperature: {LOOKBACK}-Day Lookback Comparison")
plt.xlabel("Epochs")
plt.ylabel("Validation MAE (Scaled)")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot for the assignment
plt.savefig(os.path.join(output_dir, "final_poc_comparison.png"))
print(f"\nSuccess! Check the '{output_dir}' folder for your essay artifacts.")
plt.show()
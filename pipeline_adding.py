import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# 1. SETUP
output_dir = "poc_final_stress_test"
logs_dir = os.path.join(output_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
tf.random.set_seed(42)

# 2. DATA GENERATION: The "Ading Problem"d
# This is the gold standard for proving LSTMs.
def generate_adding_data(num_samples, seq_len):
    # Random numbers between 0 and 1
    x_data = np.random.uniform(0, 1, (num_samples, seq_len, 1))
    # Indicator mask: two random positions set to 1, others 0
    mask = np.zeros((num_samples, seq_len, 1))
    for i in range(num_samples):
        indices = np.random.choice(seq_len, 2, replace=False)
        mask[i, indices] = 1
    # Features: Concatenate the random numbers and the mask
    x_features = np.concatenate((x_data, mask), axis=2)
    # Target: The sum of the two numbers indicated by the mask
    y_targets = np.sum(x_data * mask, axis=1)
    return x_features, y_targets

SEQ_LEN = 200 # At this length, SimpleRNN gradients effectively hit 0
X, y = generate_adding_data(10000, SEQ_LEN)

split = int(0.8 * len(X))
x_train, x_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# 3. EXPERIMENT ENGINE
def build_and_train(layer_type, name):
    model = tf.keras.Sequential([
        layer_type(64, input_shape=(SEQ_LEN, 2)),
        tf.keras.layers.Dense(1)
    ])
    
    # Use a solid LR for a 100-epoch run
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    callbacks = [
        tf.keras.callbacks.CSVLogger(os.path.join(logs_dir, f"{name}_log.csv")),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(output_dir, f"{name}_best.h5"), save_best_only=True)
    ]

    print(f"\n--- Training {name} for 100 Epochs ---")
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                        epochs=100, batch_size=64, callbacks=callbacks, verbose=1)
    return history

# Run the stress test
h_rnn = build_and_train(tf.keras.layers.SimpleRNN, "RNN")
h_lstm = build_and_train(tf.keras.layers.LSTM, "LSTM")

# 4. FINAL PLOT
plt.figure(figsize=(12, 6))
rnn_log = os.path.join(logs_dir, "RNN_log.csv")
lstm_log = os.path.join(logs_dir, "LSTM_log.csv")

if os.path.exists(rnn_log) and os.path.exists(lstm_log):
    r = np.loadtxt(rnn_log, delimiter=',', skiprows=1)
    l = np.loadtxt(lstm_log, delimiter=',', skiprows=1)
    
    plt.plot(r[:, 2], label='SimpleRNN (Vanishing Gradient)', color='red', lw=2)
    plt.plot(l[:, 2], label='LSTM (Gated Memory)', color='blue', lw=2)
    
    plt.title(f"Final Stress Test: Adding Problem (Sequence Length {SEQ_LEN})")
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "final_proof.png"))

print(f"\nExperiment Complete. Check '{output_dir}/final_proof.png' tomorrow.")
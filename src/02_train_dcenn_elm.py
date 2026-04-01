import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

SEED = 42

# ==========================================
# 1. dCeNN Architecture (PyTorch)
# ==========================================
class CentroidEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CentroidEncoder, self).__init__()
        # Encoder: Compresses the input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        # Decoder: Attempts to reconstruct the input (used only for training)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

# ==========================================
# 2. Extreme Learning Machine (ELM)
# ==========================================
class ELMRegressor:
    def __init__(self, input_dim, hidden_dim, reg_lambda=1e-2, random_state=SEED):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reg_lambda = reg_lambda
        self.rng = np.random.default_rng(random_state)
        # Randomly initialize fixed weights and biases
        self.W = self.rng.standard_normal((input_dim, hidden_dim))
        self.b = self.rng.standard_normal(hidden_dim)
        self.beta = None # Output weights to be calculated

    def _relu(self, x):
        return np.maximum(0, x)

    def fit(self, X, y):
        # Calculate hidden layer output matrix (H)
        H = self._relu(np.dot(X, self.W) + self.b)
        # Calculate output weights (beta) using ridge-regularized closed form
        ht_h = H.T @ H
        reg = self.reg_lambda * np.eye(self.hidden_dim)
        self.beta = np.linalg.solve(ht_h + reg, H.T @ y)

    def predict(self, X):
        H = self._relu(np.dot(X, self.W) + self.b)
        return np.dot(H, self.beta)

# ==========================================
# 3. Execution Pipeline
# ==========================================
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading preprocessed data...")
    df = pd.read_csv("data/processed_15min.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Split: Train on 2023, Test on 2024
    train_df = df[df['timestamp'] < '2024-01-01']
    test_df = df[df['timestamp'] >= '2024-01-01']

    # Define features (X) and target (y)
    # Best configuration from full benchmark sweep (feature_set: baseline_plus_calendar)
    # RMSE=10.02, MAE=7.42, ASP anomalies=42 — best accuracy AND safety across 128 runs
    feature_cols = [
        'cglo', 'ffam', 'rr', 'tl', 'time_sin', 'time_cos',
        'month_sin', 'month_cos', 'hour', 'day_of_week', 'is_holiday',
        'power_lag_15m', 'power_lag_1h', 'power_lag_6h', 'power_lag_24h', 'power_lag_48h',
        'power_roll_mean_6h', 'power_roll_std_6h', 'power_roll_mean_24h',
        'power_delta_15m', 'power_delta_1h',
        'cglo_temp_interaction', 'wind_rain_interaction', 'cglo_squared',
    ]
    target_col = 'power_generation'

    available = [c for c in feature_cols if c in df.columns]
    missing = sorted(set(feature_cols) - set(available))
    if missing:
        print(f"Warning: missing engineered features, continuing with available columns: {missing}")
    feature_cols = available

    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test_raw = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Scaling: Essential for Neural Networks
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)

    # --- Step A: Train the dCeNN ---
    print("Training dCeNN (Centroid Encoder)...")
    input_dim = X_train_scaled.shape[1]
    latent_dim = min(20, input_dim)  # Best-performing search result on this dataset

    autoencoder = CentroidEncoder(input_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0025)

    # Quick training loop for the autoencoder
    epochs = 260
    for epoch in range(epochs):
        optimizer.zero_grad()
        latent, reconstructed = autoencoder(X_train_tensor)
        loss = criterion(reconstructed, X_train_tensor)
        loss.backward()
        optimizer.step()

    print(f"dCeNN Training Complete. Final Autoencoder Loss: {loss.item():.4f}")

    # Extract latent features (freeze the network, no gradients needed)
    with torch.no_grad():
        Z_train, _ = autoencoder(X_train_tensor)
        Z_test, _ = autoencoder(X_test_tensor)
    
    Z_train_np = Z_train.numpy()
    Z_test_np = Z_test.numpy()

    # --- Step B: Train the ELM ---
    print("Training Extreme Learning Machine (ELM)...")
    elm_hidden_neurons = 2048
    ensemble_size = 11
    reg_lambda = 1e-3

    models = []
    for i in range(ensemble_size):
        elm = ELMRegressor(
            input_dim=latent_dim,
            hidden_dim=elm_hidden_neurons,
            reg_lambda=reg_lambda,
            random_state=SEED + i,
        )
        elm.fit(Z_train_np, y_train)
        models.append(elm)

    # --- Step C: Evaluation ---
    print("Generating Predictions...")
    member_predictions = np.column_stack([m.predict(Z_test_np) for m in models])
    predictions = member_predictions.mean(axis=1)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.nanmean(np.abs((y_test - predictions) / np.where(y_test == 0, np.nan, y_test))) * 100
    nrmse_mean = rmse / np.mean(y_test)
    nrmse_range = rmse / (np.max(y_test) - np.min(y_test))
    cv_rmse = rmse / np.std(y_test)
    r2 = r2_score(y_test, predictions)

    print("\n" + "="*30)
    print("MODEL EVALUATION (2024 Test Set)")
    print("="*30)
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE:  {mae:.2f} MW")
    print(f"MAPE: {mape:.2f} %")
    print(f"NRMSE (mean): {nrmse_mean * 100:.2f} %")
    print(f"NRMSE (range): {nrmse_range * 100:.2f} %")
    print(f"CV-RMSE: {cv_rmse * 100:.2f} %")
    print(f"R2: {r2:.4f}")
    print("="*30)

    # Save predictions for the ASP reasoning phase
    test_df = test_df.copy()
    test_df['predicted_generation'] = predictions
    test_df.to_csv("data/predictions_2024.csv", index=False)
    print("Saved predictions to data/predictions_2024.csv")

if __name__ == "__main__":
    main()
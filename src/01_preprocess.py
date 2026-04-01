import pandas as pd
import numpy as np
import os
import holidays

def load_and_align_data(gen_path, weather_path):
    print("Loading datasets...")
    # 1. Load the data
    df_gen = pd.read_excel(gen_path, sheet_name='data_2023')
    df_weather = pd.read_csv(weather_path)
    
    # 2. Standardize Column Names
    df_gen.rename(columns={'Date': 'timestamp', 'Value': 'power_generation'}, inplace=True)
    
    # 3. Handle Timezones and Datetimes
    # Weather is in UTC (+00:00). Generation is likely local Austrian time.
    # We will convert weather to local Austrian time (Europe/Vienna) to align them.
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_weather['timestamp'] = df_weather['timestamp'].dt.tz_convert('Europe/Vienna').dt.tz_localize(None)
    
    df_gen['timestamp'] = pd.to_datetime(df_gen['timestamp'])
    
    # 4. Merge Datasets
    print("Merging on 15-minute intervals...")
    df_merged = pd.merge(df_weather, df_gen, on='timestamp', how='inner')
    
    # Sort chronologically just in case
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
    return df_merged

def create_features(df):
    print("Engineering cyclical and lag features...")
    # Extract basic time attributes
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Time of day in decimal hours (e.g., 14:15 -> 14.25)
    time_decimal = df['hour'] + df['minute'] / 60.0
    
    # Cyclical Transformations (Sine/Cosine)
    # This teaches the AI that 23:45 and 00:00 are close to each other
    df['time_sin'] = np.sin(2 * np.pi * time_decimal / 24.0)
    df['time_cos'] = np.cos(2 * np.pi * time_decimal / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Austrian Public Holidays feature
    years = df['timestamp'].dt.year.unique().tolist()
    at_holidays = holidays.Austria(years=years)
    df['is_holiday'] = df['timestamp'].dt.date.map(lambda d: 1 if d in at_holidays else 0)

    # Autoregressive lag features at multiple horizons
    # 1 step=15m, 4 steps=1h, 24 steps=6h, 96 steps=24h, 192 steps=48h
    df['power_lag_15m'] = df['power_generation'].shift(1)
    df['power_lag_1h'] = df['power_generation'].shift(4)
    df['power_lag_6h'] = df['power_generation'].shift(24)
    df['power_lag_24h'] = df['power_generation'].shift(96)
    df['power_lag_48h'] = df['power_generation'].shift(192)

    # Rolling statistics from past observations only (shifted by 1 to avoid leakage)
    shifted_power = df['power_generation'].shift(1)
    df['power_roll_mean_6h'] = shifted_power.rolling(window=24).mean()
    df['power_roll_std_6h'] = shifted_power.rolling(window=24).std()
    df['power_roll_mean_24h'] = shifted_power.rolling(window=96).mean()

    # Recent change features
    df['power_delta_15m'] = df['power_generation'].diff(1)
    df['power_delta_1h'] = df['power_generation'].diff(4)

    # Weather interaction terms to capture non-linear production dynamics
    df['cglo_temp_interaction'] = df['cglo'] * df['tl']
    df['wind_rain_interaction'] = df['ffam'] * df['rr']
    df['cglo_squared'] = df['cglo'] ** 2
    
    # Drop rows with NaN values created by the lag shift
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    # Define file paths
    RAW_GEN_PATH = "data/gen_dataset.xlsx"
    RAW_WEATHER_PATH = "data/weather_data_15min.csv"
    OUTPUT_PATH = "data/processed_15min.csv"
    
    # Execute pipeline
    merged_data = load_and_align_data(RAW_GEN_PATH, RAW_WEATHER_PATH)
    processed_data = create_features(merged_data)
    
    # Save the ready-to-train dataset
    processed_data.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessing complete. Saved {len(processed_data)} rows to {OUTPUT_PATH}")
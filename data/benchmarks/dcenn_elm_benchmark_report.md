# dCeNN-ELM Benchmark Report

## Best Configuration

- Feature set: baseline_plus_calendar
- Feature count: 11
- Latent dim: 10
- ELM hidden neurons: 256
- Epochs: 50
- Learning rate: 0.01
- RMSE: 9.9447
- MAE: 7.4357
- MAPE: 9.27%
- Evening MAE (18-23): 10.1321
- ASP anomalies in test window: 48

## Top 10 Configurations by RMSE

| feature_set | latent_dim | elm_hidden_neurons | epochs | lr | rmse | mae | mape% | asp_anomalies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_plus_calendar | 10 | 256 | 50 | 0.01 | 9.9447 | 7.4357 | 9.27 | 48 |
| baseline_plus_minute | 10 | 256 | 50 | 0.01 | 10.2945 | 7.5633 | 9.31 | 62 |
| baseline | 8 | 256 | 50 | 0.01 | 10.6048 | 7.6885 | 9.24 | 56 |
| baseline_plus_calendar | 10 | 128 | 50 | 0.01 | 10.7218 | 8.1888 | 10.51 | 68 |
| baseline_plus_calendar | 8 | 256 | 50 | 0.01 | 10.8539 | 7.9373 | 9.65 | 72 |
| baseline | 10 | 128 | 50 | 0.01 | 10.9457 | 7.9829 | 9.64 | 72 |
| baseline_plus_minute | 8 | 256 | 50 | 0.01 | 10.9676 | 8.0927 | 9.88 | 54 |
| baseline | 8 | 128 | 50 | 0.01 | 11.0033 | 8.0166 | 9.69 | 71 |
| baseline_plus_minute | 10 | 128 | 50 | 0.01 | 11.2524 | 8.5126 | 10.65 | 81 |
| baseline_plus_calendar | 8 | 128 | 50 | 0.01 | 11.3263 | 8.4046 | 10.34 | 71 |

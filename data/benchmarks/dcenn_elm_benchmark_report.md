# dCeNN-ELM Benchmark Report

## Best Configuration

- Feature set: autoregressive_enhanced
- Feature count: 24
- Latent dim: 20
- ELM hidden neurons: 2048
- Epochs: 260
- Learning rate: 0.0025
- RMSE: 3.6405
- MAE: 2.6236
- MAPE: 3.26%
- Evening MAE (18-23): 3.5546
- ASP anomalies in test window: 14

## Top 10 Configurations by RMSE

| feature_set | latent_dim | elm_hidden_neurons | epochs | lr | rmse | mae | mape% | asp_anomalies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| autoregressive_enhanced | 20 | 2048 | 260 | 0.0025 | 3.6405 | 2.6236 | 3.26 | 14 |

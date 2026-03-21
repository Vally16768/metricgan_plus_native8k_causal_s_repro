# MetricGAN+ native8k causal_s repro report

- Model final: `metricgan_plus_native8k_causal_s-small-lr0.0002-seg16000-lossD2-seed0`
- Run de referinta: `2a2b1b3f2c7c4f1f9087a23e2dfdf762`
- Castigator stage1: `metricgan_plus_native8k_causal_s-small-lr0.0005-seg16000-lossD1-seed0`
- Best val_select PESQ: `2.8608383980023087`
- Test PESQ: `3.327003804804052`
- Test STOI: `0.9334712175333594`
- Test SI-SDR: `17.918519973754883`
- Test Delta SNR: `9.569960514669376`
- Benchmark latency 10s: `0.0472748503331483`
- Deployable profiles: `['stm32u5_low_power_rt', 'nrf54h20_low_power_rt', 'apollo4_blue_plus_low_power_rt', 'alif_ensemble_e3_ai_audio_rt', 'imx_rt700_ai_audio_rt', 'alif_ensemble_e6_ai_audio_rt', 'stm32n6_ai_audio_rt', 'ra8p1_ai_audio_rt']`
- Profiles <50mW: `['stm32u5_low_power_rt', 'nrf54h20_low_power_rt', 'apollo4_blue_plus_low_power_rt', 'alif_ensemble_e3_ai_audio_rt', 'imx_rt700_ai_audio_rt', 'alif_ensemble_e6_ai_audio_rt', 'stm32n6_ai_audio_rt', 'ra8p1_ai_audio_rt']`

## Artefacte

- `canonical_metrics.csv`: sumar metrici canonice si de evaluare
- `metric_history.csv`: istoric complet extras din run-urile originale
- `training_curves.png`: curbele de antrenare/QAT
- `stage1_comparison.png`: comparatie intre candidatii stage1
- `deployability_profiles.png`: sumar simulare MCU
- `sample_waveforms.png` si `sample_spectrograms.png`: audit pe sample audio

## Observatie

DNSMOS si metricele necanonice raman optionale si nu sunt generate implicit in acest proiect.

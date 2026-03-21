# Proiect Standalone Pentru `metricgan_plus_native8k_causal_s-small-lr0.0002-seg16000-lossD2-seed0`

Acest proiect izoleaza strict linia de antrenare si evaluare pentru modelul deployable:

- teacher sursa: `metricgan_plus_native8k_small.pt`
- stage1: `metricgan_plus_native8k_causal_s` si `metricgan_plus_native8k_causal_xs`, seed `0/1`, loss `D1`
- QAT final: `metricgan_plus_native8k_causal_s`, loss `D2`, seed-ul castigatorului stage1
- evaluare finala: `val_rank`, `val_select`, `test`
- simulare deployability MCU: local, pe shortlist-ul din `sebench/stm32sim.py`

Proiectul este standalone la runtime: nu importa module din `ULP-SE-aTENNuate`. Pentru bootstrap-ul initial am inclus doar un script separat care copiaza checkpoint-uri si exporta istoriile reale ale run-urilor originale in `reference/`.

## Structura

- `repro.py`: CLI-ul principal
- `configs/default.yaml`: configuratia unica a pipeline-ului
- `sebench/`: nucleul copiat/simplificat pentru acest model
- `scripts/export_reference_runs.py`: export local al celor 5 run-uri originale si al checkpoint-urilor de referinta
- `reference/`: checkpoint-uri si istorii bootstrap
- `outputs/`: evaluari, rapoarte, checkpoint-uri noi si tracking local

## Rezultate obținute

Proiectul a fost evaluat pe seturile VoiceBank-DEMAND la 8kHz, cu rezultate excelente pentru îmbunătățirea calității vorbirii:

### Versiunea QAT (quantized aware training):
- **PESQ test:** 3.327
- **STOI test:** 0.933
- **SI-SDR test:** 17.919 dB
- **Delta SNR test:** 9.570 dB
- **Latență benchmark (10s audio):** 0.0037 secunde
- **Profile MCU suportate:** STM32U5, NRF54H20, Apollo4 Blue Plus, IMX RT700, STM32N6, RA8P1 (toate sub 50mW)

### Versiunea full (nequantizată):
- **PESQ test:** 3.327
- **STOI test:** 0.933
- **SI-SDR test:** 17.919 dB
- **Delta SNR test:** 9.570 dB
- **Latență benchmark:** 0.047 secunde
- **Profile MCU suplimentare:** Include și Alif Ensemble E3/E6

Modelele sunt optimizate pentru deployare pe microcontrolere cu consum scăzut de energie.

## Configurație actualizată

Dataset-ul este configurat să folosească samples din locația externă `/home/vali/Desktop/Disc-1TB/ULP-SE-aTENNuate/dataset/voicebank-demand` fără să le copieze local în proiect. Acest lucru economisește spațiu și permite partajarea datelor între proiecte.

Căile principale în `configs/default.yaml`:
- `dataset_root`: `/home/vali/Desktop/Disc-1TB/ULP-SE-aTENNuate/dataset/voicebank-demand`
- `source_mlruns_root`: `/home/vali/Desktop/Disc-1TB/ULP-SE-aTENNuate/runtime/mlruns`

Proiectul folosește virtual environment-ul partajat la `/home/vali/Desktop/ULP/shared-venv` pentru execuție.

## Comenzi

Interpreter recomandat pentru verificare rapida in mediul actual:

```bash
/home/vali/Desktop/ULP/shared-venv/bin/python
```

Comenzile principale:

```bash
python repro.py prepare_data
python repro.py build_teacher_cache
python repro.py train_stage1 --device cuda
python repro.py train_qat --device cuda
python repro.py evaluate --device auto
python repro.py report
python repro.py run_all --device cuda
```

Bootstrap de referinta:

```bash
python scripts/export_reference_runs.py
```

## Configuratie implicita

`configs/default.yaml` fixeaza hiperparametrii reali pentru run-ul original:

- `sample_rate=8000`
- `n_fft=256`
- `hop_length=80`
- `win_length=160`
- `segment_len=16000`
- `erb_bands=32`
- `context_frames=5`
- `guidance_classic=none`
- stage1: `D1`, `lr=5e-4`, `epochs=100`, `early_stop_patience=8`, `min_epochs=15`
- QAT: `D2`, `lr=2e-4`, `epochs=20`, `early_stop_patience=4`, `min_epochs=10`
- selectie finala: `best/val_select_pesq_mean`

`batch_size`, `grad_accum` si `eval_batch_size` sunt lasate sa urmeze logica originala din `sebench.training.apply_runtime_profile`.

## Split-uri VoiceBank + DEMAND

`prepare_data` reproduce exact split-ul intern de campanie:

- `val_speakers = p239, p286, p244, p270`
- `rank_count = 128`

Manifestul `8k/test.csv` este materializat explicit, dar pastreaza aceleasi perechi oficiale de test.

## Teacher Cache

`build_teacher_cache`:

1. incarca `metricgan_plus_native8k_small.pt`
2. aplica cuantizare dinamica locala
3. genereaza `teacher_lite_cache_8k/train_fit/train_fit_teacher_cache.csv`
4. salveaza `teacher_wav`, `teacher_mask_erb` si optional guidance

## Tracking si rapoarte

Tracking-ul este local, in `tracking/`, fara MLflow extern obligatoriu. Formatul retine:

- `params.json`
- `latest_metrics.json`
- `metrics_history.jsonl`
- `artifacts/`

`report` construieste:

- `canonical_metrics.csv`
- `metric_history.csv`
- `report_summary.json`
- `report.md`
- `training_curves.png`
- `stage1_comparison.png`
- `deployability_profiles.png`
- `sample_waveforms.png`
- `sample_spectrograms.png`
- `audio_samples/`

## Metrici canonice

Raportul include explicit:

- `train/loss`
- `train/wave_loss`
- `train/spectral_loss`
- `train/sisdr_loss`
- `train/teacher_mask_loss`
- `train/teacher_wave_loss`
- `lr`
- `val_rank/pesq_mean`
- `val_rank/stoi_mean`
- `val_rank/sisdr_mean`
- `val_rank/delta_snr_mean`
- `best/val_select_pesq_mean`
- `best/val_select_stoi_mean`
- `best/val_select_sisdr_mean`
- `best/val_select_delta_snr_mean`
- `test/pesq_mean`
- `test/stoi_mean`
- `test/sisdr_mean`
- `test/delta_snr_mean`
- `benchmark_latency_10s`
- metrici de deployability/simulator pe shortlist-ul MCU

### Metrice suplimentare (implicite în evaluare)

- `count`
- `csig_mean`
- `cbak_mean`
- `covl_mean`
- `dnsmos_sig_mean`
- `dnsmos_bak_mean`
- `dnsmos_ovr_mean`

Acestea sunt generate de fiecare dată când rulezi evaluarea, pentru testare și comparare cross-dataset.

Metricele necanonice precum `DNSMOS`, `CSIG`, `CBAK`, `COVL` raman optionale. In configuratia curenta, `DNSMOS` este oprit implicit, exact ca in run-ul original.

## Lineage si tolerante

Linia reproducerii este:

1. `metricgan_plus_native8k_small.pt`
2. cache `teacher_lite_cache_8k`
3. stage1 `causal_s` si `causal_xs`
4. selectia castigatorului dupa `best/val_select_pesq_mean`
5. QAT final `D2`
6. evaluare completa pe `val_rank`, `val_select`, `test`

Tolerante recomandate:

- pentru evaluarea checkpoint-ului de referinta: practic identic cu exportul original
- pentru rerulare integrala de training: apropiere numerica rezonabila, nu identitate bitwise

## Validare recomandata

Ordinea minima de verificare:

```bash
python scripts/export_reference_runs.py
python repro.py prepare_data
python repro.py evaluate --device auto
python repro.py report
```

Pentru rerulare completa a antrenarii este recomandat `--device cuda`.

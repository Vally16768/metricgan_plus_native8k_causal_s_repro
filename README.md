# Proiect Standalone Pentru `metricgan_plus_native8k_causal_s-small-lr0.0002-seg16000-lossD2-seed0`

Acest proiect izoleaza linia canonica deployable pentru `VoiceBank+DEMAND 8 kHz`:

- teacher mare: `metricgan_plus_native8k_small.pt`
- student castigator: `metricgan_plus_native8k_causal_s`
- `stage1` cu distilare din teacher cache
- `QAT` final cu reteta `D2`
- evaluare pe `val_rank`, `val_select`, `test`
- simulare de deployability MCU pe shortlist-ul din `sebench/stm32sim.py`

Proiectul este standalone la runtime: nu importa module din `ULP-SE-aTENNuate`.

Pentru documentatia tehnica exhaustiva in romana despre dataset, arhitectura, distilare, QAT, selectie, rezultate si comparatii, vezi [README_extended.md](README_extended.md).

## Scope

Acest `README.md` ramane intentionat scurt si operational. El documenteaza doar linia finala `8 kHz` folosita pentru modelul deployable curent.

Configuri alternative precum `DNS5`, `combined` sau `reference_scenario` exista in `configs/`, dar nu sunt fluxul primar al acestui README.

## Rezultatul Curent

Modelul final `QAT` are urmatoarele rezultate locale pe `VoiceBank+DEMAND 8 kHz`:

- `val_select PESQ = 2.8608`
- `test PESQ = 3.3269`
- `test STOI = 0.9335`
- `test SI-SDR = 17.9188 dB`
- `test Delta SNR = 9.5703 dB`
- `benchmark_latency_10s = 0.0036808 s`

Profile MCU suportate in shortlist:

- `STM32U5`
- `nRF54H20`
- `Apollo4 Blue+`
- `i.MX RT700`
- `STM32N6`
- `RA8P1`

## Mediul Curent

Cai importante din `configs/default.yaml`:

- `dataset_root = /home/vali/Desktop/Disc-1TB/ULP-SE-aTENNuate/dataset/voicebank-demand`
- `source_mlruns_root = /home/vali/Desktop/Disc-1TB/ULP-SE-aTENNuate/runtime/mlruns`
- `reference_root = {project_root}/reference`
- `output_root = {project_root}/outputs`

Interpreter recomandat in mediul actual:

```bash
/home/vali/Desktop/ULP/shared-venv/bin/python
```

## Quick Start

### 1. Bootstrap referinta

```bash
python scripts/export_reference_runs.py
```

Aceasta comanda copiaza checkpoint-urile si exporta istoria run-urilor de referinta in `reference/`.

### 2. Verificare minima fara rerulare completa de training

```bash
python repro.py prepare_data
python repro.py evaluate --device auto
python repro.py report
```

Aceasta este ordinea minima recomandata pentru a reconstrui artefactele de evaluare si raport pentru linia canonica.

### 3. Rerulare completa a liniei deployable

```bash
python repro.py prepare_data
python repro.py build_teacher_cache --device cuda
python repro.py train_stage1 --device cuda
python repro.py train_qat --device cuda
python repro.py evaluate --device cuda
python repro.py report
```

Pentru rerulare completa este recomandat `--device cuda`.

## Configuratia Canonica

`configs/default.yaml` fixeaza hiperparametrii liniei originale:

- `sample_rate = 8000`
- `n_fft = 256`
- `hop_length = 80`
- `win_length = 160`
- `segment_len = 16000`
- `erb_bands = 32`
- `context_frames = 5`
- `guidance_classic = none`
- `stage1 = D1, lr=5e-4, epochs=100`
- `qat = D2, lr=2e-4, epochs=20`
- selectie finala intre candidati dupa `best/val_select_pesq_mean`

Splitul de campanie pentru `VoiceBank+DEMAND 8 kHz` este:

- `val_speakers = p239, p286, p244, p270`
- `rank_count = 128`

## Structura Relevanta

- `repro.py`: CLI principal pentru pipeline
- `configs/default.yaml`: configul canonical `8 kHz`
- `sebench/`: implementarea standalone pentru modele, training, cache, splituri si raportare
- `reference/`: checkpoint-uri si istoric bootstrap
- `outputs/`: evaluari, rapoarte si checkpoint-uri locale
- `tracking/`: tracking local fara MLflow extern obligatoriu

## Artefacte Generate

`report` construieste in mod normal:

- `outputs/evaluations/reference_qat/`
- `outputs/reports/reference_qat/report.md`
- `outputs/reports/reference_qat/report_summary.json`
- `outputs/reports/reference_qat/training_curves.png`
- `outputs/reports/reference_qat/stage1_comparison.png`
- `outputs/reports/reference_qat/deployability_profiles.png`
- `outputs/reports/reference_qat/sample_waveforms.png`
- `outputs/reports/reference_qat/sample_spectrograms.png`

## Note Operationale

- `build_teacher_cache` salveaza `teacher_wav` si `teacher_mask_erb` pentru train.
- `val_rank` este folosit pentru selectie in interiorul unui run.
- `val_select` este folosit pentru comparatia finala intre candidati.
- `test` ramane hold-out final.
- Toleranta corecta la rerulare completa este apropiere numerica rezonabila, nu identitate bitwise.

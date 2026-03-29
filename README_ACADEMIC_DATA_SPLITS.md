# Academic Data Split Policy (VoiceBank+DEMAND + DNS5)

## Scope

Acest document descrie decizia de split pentru antrenarea:

- teacher `MetricGAN+`
- student `metricgan_plus_native8k_causal_s`

pe dataset combinat `VoiceBank+DEMAND + DNS5`, cu reguli anti-leakage.

## Decizia academică adoptată

1. `VoiceBank+DEMAND` păstrează split-ul oficial:
- train: `voicebank-demand/16k/train.csv`
- test: `voicebank-demand/16k/test.csv`

2. Validation pentru `VoiceBank+DEMAND` se construiește doar din train, prin speaker holdout:
- speakers validați: `p239`, `p286`, `p244`, `p270`
- manifesteri: `train_fit`, `val_rank`, `val_select` în `voicebank-demand/16k/campaign/`

3. `DNS5` local nu are test oficial separat în acest workspace.
- se folosește `dns5-headset-16k/train.csv` + `dns5-headset-16k/val.csv`
- dacă `dns5_test_csv` lipsește, se construiește deterministic test din `10%` din train:
- `dns5_test_from_train_fraction: 0.10`
- `split_seed: 42`

4. `DNS5 val` este împărțit disjunct:
- `dns5_val_rank.csv`
- `dns5_val_select.csv`

5. Combined manifests:
- `train_combined = VoiceBank train_fit + DNS5 train_fit`
- `val_rank_combined = VoiceBank val_rank + DNS5 val_rank`
- `val_select_combined = VoiceBank val_select + DNS5 val_select`
- `test_combined = VoiceBank official test + DNS5 test (derivat sau explicit)`

## Anti-leakage și integritate

Mecanisme implementate:

- deduplicare exactă perechi `(noisy, clean)` la concatenare
- audit de overlap pe `pair_key` și `clean_key` la `prepare_data`
- blocare runtime în training dacă există:
- duplicate în manifest
- overlap `train vs val`
- overlap `train vs test`
- overlap `val vs test`

Fișiere implicate:

- `repro.py`:
- `dataset_type: academic_combined`
- split DNS5 train/test fallback (10%)
- split DNS5 val în rank/select
- audit integritate combined

- `sebench/training.py`:
- validare integritate la start de run (`_validate_manifest_integrity`)

- `sebench/losses.py`:
- `T0` pentru teacher training fără teacher-cache

## Rezultatul curent (după `prepare_data --force`)

Sursă: `outputs/combined_datasets/prepare_data/summary.json`

- `train_combined`: `355,521`
- `val_rank_combined`: `4,224`
- `val_select_combined`: `28,472`
- `test_combined`: `39,242`

DNS5 derivat din train:

- `dns5_train`: `384,185`
- `dns5_train_fit`: `345,767`
- `dns5_test_derived`: `38,418` (10%)

Integritate:

- duplicate perechi: `0`
- duplicate clean keys: `0`
- overlap train/val: `0`
- overlap train/test: `0`
- overlap val/test: `0`

Raport:

- `outputs/combined_datasets/combined/combined_integrity_summary.json`

## Config activ

Fișier: `configs/scenario_combined_datasets.yaml`

Setări relevante:

- `dataset_type: academic_combined`
- `dns5_test_from_train_fraction: 0.10`
- `split_seed: 42`
- `val_speakers: [p239, p286, p244, p270]`

## Reproducere

```bash
cd /home/vali/Desktop/ULP/metricgan_plus_native8k_causal_s_repro
/home/vali/Desktop/ULP/shared-venv/bin/python3 repro.py --config configs/scenario_combined_datasets.yaml prepare_data --force --device auto
```

## Teacher + Student pe aceleași split-uri

Teacher și student consumă aceleași manifeste combined din config:

- train: `combined/train_combined.csv`
- val_rank: `combined/val_rank_combined.csv`
- val_select: `combined/val_select_combined.csv`
- test: `combined/test_combined.csv`

## Când apare test oficial DNS5

Dacă obții manifest oficial DNS5 test/dev:

- setezi `dns5_test_csv` în config
- fallback-ul `10% din train` nu mai este folosit
- pipeline-ul include automat testul oficial în `test_combined`

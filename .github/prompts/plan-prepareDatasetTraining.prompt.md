Scenarii de training pentru proiect:

1) VoiceBank+DEMAND 16k
   - dataset_root: /home/vali/Desktop/Disc-1TB/ULP-SE-aTENNuate/dataset/voicebank-demand
   - train_csv_16k: {dataset_root}/16k/train.csv
   - test_csv_16k: {dataset_root}/16k/test.csv
   - campaign_dir_8k: {dataset_root}/8k/campaign
   - train_fit_csv: {dataset_root}/8k/campaign/train_fit.csv
   - val_rank_csv: {dataset_root}/8k/campaign/val_rank.csv
   - val_select_csv: {dataset_root}/8k/campaign/val_select.csv
   - test_csv_8k: {dataset_root}/8k/test.csv
   - output_root: {project_root}/outputs/vbd16k
   - tracking_root: {project_root}/tracking/vbd16k
   - stage1/qat results separate

2) VoiceBank+DEMAND 8k
   - dataset_root: /home/vali/Desktop/Disc-1TB/ULP-SE-aTENNuate/dataset/voicebank-demand
   - train_csv_16k: not needed or stocat pe 8k se trimite direct
   - test_csv_16k: not needed
   - campaign_dir_8k: {dataset_root}/8k/campaign
   - train_fit_csv/val_rank_csv/val_select_csv/test_csv_8k la fel ca mai sus
   - option: rule this as sample_rate 8000 direct, plus `dataset_type: voicebank8k`

3) DNS5-headset-16k
   - dataset_root: /home/vali/Desktop/Disc-1TB/ULP-SE-aTENNuate/dataset/dns5-headset-16k
   - train_csv_16k: {dataset_root}/train.csv
   - test_csv_16k: {dataset_root}/test.csv
   - nu necesita campanie 8k, `dataset_type: dns5`
   - val split: din csv sau reuse test/val; dacă nu exista se fac script separat (split 80/10/10)
   - output_root: {project_root}/outputs/dns5_16k
   - tracking_root: {project_root}/tracking/dns5_16k

4) DNS5-headset-16k + VoiceBank+DEMAND-16k (mix)
   - dataset_root: combined, sau foloseste două root-uri distincte
   - train manifest combinat: {root_dns5}/train.csv + {root_vbd}/16k/train.csv (concat)
   - test util: se face un manifest combinat din {root_dns5}/test.csv + {root_vbd}/16k/test.csv
   - val_rank/val_select: opțional mix sau 1 set dependent, dar se poate face opțional 
   - output_root: {project_root}/outputs/dns5_vbd16k
   - tracking_root: {project_root}/tracking/dns5_vbd16k

Acțiuni necesare în `repro.py`:
- Adaugă `command_prepare_dataset(config, *, force=False)`:
   - `dataset_type: voicebank` -> `build_voicebank_campaign_splits` + `materialize_test_manifest_8k`
   - `dataset_type: dns5` -> 1. confirmă existența CSV-uri predefinite 16k; 2. populare direct `train_csv_16k/test_csv_16k` și `test_csv`;
   - `dataset_type: combined` -> construiește CSV combinat pentru train/test și val (dacă e nevoie)
- Retușează `command_train_stage1` / `command_train_qat` să folosească `config["paths"]["output_root"]` și `config["paths"]["tracking_root"]` specifice scenariului.
- Asigură `checkpoint_out` distinct:  `{output_root}/checkpoints/stage1/<run_name>.pt` și `{output_root}/checkpoints/qat/<run_name>.pt`.
- Când rulezi `command_evaluate`, setare `config["evaluation"]["test_csv"]` la CSVul test per scenariu. Evaluare cross:
   - `dns5->vbd`: `config` cu model din `outputs/dns5_16k/checkpoints/...` și test_csv voicebank test combined
   - `vbd->dns5`: invers.

Config YAML: creează:
- `configs/scenario_vbd_16k.yaml`
- `configs/scenario_vbd_8k.yaml`
- `configs/scenario_dns5_16k.yaml`
- `configs/scenario_dns5_vbd_16k.yaml`
- `configs/scenario_cross_dns5_to_vbd.yaml`
- `configs/scenario_cross_vbd_to_dns5.yaml`

Script de orchestrare (ex. `scripts/run_scenarios.sh`):
1. `python repro.py prepare_dataset --config configs/scenario_vbd_16k.yaml --force`
2. `python repro.py train_stage1 --config configs/scenario_vbd_16k.yaml --device cuda`
3. `python repro.py train_qat --config configs/scenario_vbd_16k.yaml --device cuda`
4. `python repro.py evaluate --config configs/scenario_vbd_16k.yaml --device auto`
5. analog pentru celelalte 3 scenarii
6. cross-evaluate cu scenariile post-training

Output separat asigurat prin `output_root`+`tracking_root` scenariu.

Remark: asigură `configs/default.yaml` păstrează baza folositoare (val_speakers, rank_count, sample_rate etc.).

Verificare finală:
- `find outputs -type f | grep checkpoint` produce 4 subfoldere: vbd16k, vbd8k, dns5_16k, dns5_vbd16k.
- `find outputs -type f | grep report_summary.json` produce 4 rapoarte.

Următorul pas: implementăm patch-ul direct în fișiere (repro.py + config) și verificăm cu comanda `python repro.py prepare_dataset ...` + mini-și. 

count, pesq_mean, stoi_mean, sisdr_mean, delta_snr_mean, csig_mean, cbak_mean, covl_mean, dnsmos_sig_mean, dnsmos_bak_mean, dnsmos_ovr_mean

deci toate acestea sunt implementate. actualizeaza documentatia ca sa se stie ca de fiecare data cand rulam o evaluare ne intereseasa toate aceste metrici.
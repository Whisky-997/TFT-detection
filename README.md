 Deployment Guide — WaveLight-TFT Intrusion Detection System

This document describes how to deploy and run the TFT-based network intrusion detection
system on the **CIC-IDS-2017** dataset.

1. System Overview

The system compares several Temporal Fusion Transformer (TFT) variants for binary
intrusion detection (normal / attack):

- **LightTFT** (`LightTFT.py`) — lightweight baseline using a custom linear-attention
  module (O(N) complexity).
- **WaveLightTFT** (`wavelighttft.py` + `tft_core.py`) — the proposed model that augments
  LightTFT with a discrete-wavelet (DWT) branch and a fusion module, using standard
  multi-head attention.
- **Auxiliary experiments** — wavelet hyper-parameter search (`03小波/`), ablation studies
  and knowledge distillation (`02轻量化/`).

**I/O spec**: input flow statistics of shape `[batch, seq_len=32, input_dim=150]`
(30 base flow features × 5 window statistics); output is a binary logit.

 Environment Requirements

- **OS**: Linux recommended (developed on AutoDL Ubuntu). The model scripts also run on
  Windows, but the preprocessing paths are Linux-style absolute paths.
- **Python**: 3.9 – 3.11
- **GPU**: NVIDIA CUDA GPU strongly recommended; CPU fallback is supported but slow.
- **Dataset**: CIC-IDS-2017 — download separately from the University of New Brunswick
  (https://www.unb.ca/cic/datasets/ids-2017.html).

 Python dependencies

| Package       | Purpose                                  |
|---------------|------------------------------------------|
| torch         | model definition / training              |
| numpy         | tensor I/O                               |
| pandas        | CSV preprocessing                        |
| scikit-learn  | metrics, scalers                         |
| matplotlib    | plotting                                 |
| seaborn       | confusion-matrix heatmaps                |
| tqdm          | progress bars                            |
| thop          | FLOPs / Params profiling                 |
| PyWavelettes  | discrete wavelet transform (pywt)        |
| joblib        | scaler persistence                       |
| optuna        | hyper-parameter search (02轻量化 only)    |

Install (install PyTorch from https://pytorch.org matching your CUDA version first):

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm thop PyWavelets joblib optuna
```

 3. Repository Structure

```
TFT/
├── tft_core.py          # Core library: GRN, weight/fusion modules, LightTFTv2_1, data loader
├── LightTFT.py          # LightTFT (linear attention) — baseline
├── wavelighttft.py      # WaveLightTFT — proposed wavelet-enhanced model
├── 01预处理/             # Preprocessing pipeline (raw CSV -> TFT tensors)
│   ├── 02_clean_cic2017.py
│   ├── 04_feature_cic2017.py
│   ├── 10_standardize_data.py
│   └── 11_convert_to_tft_tensor.py
├── 02轻量化/             # Lightweight evolution / ablation / distillation (needs tft_binary.py)
└── 03小波/               # Wavelet hyper-parameter search (basis/level/fusion/position/weighting)
```

 4. Data Preparation

 4.1 Obtain the dataset

Download the CIC-IDS-2017 CSV files and place them in a raw-data directory, e.g.
`/root/autodl-tmp/graduate-thesis/data/cic2017_raw/`.

 4.2 Configure paths

Every script hardcodes absolute paths rooted at `/root/autodl-tmp/graduate-thesis/`.
Two options:

- **(A) Reproduce the default layout** — create the same directory tree on your machine.
- **(B) Adapt the paths** — edit the `BASE_DIR` / `DATA_ROOT` / `wavelet_cache_dir`
  constants at the top of each script to match your environment.

Expected data layout after preprocessing:

```
graduate-thesis/data/
├── cic2017_raw/                     # raw CSVs (input to step 1)
├── cleaned/                         # step 1 output
├── feature/                         # step 2 output + scalers/
├── standardized/                    # step 3 output + scalers/
└── tensor_T32/                      # step 4 output (model input)
    ├── cic17_W32_S16_train_X_T32.npy
    ├── cic17_W32_S16_train_y_T32.npy
    ├── cic17_W32_S16_val_X_T32.npy
    ├── cic17_W32_S16_val_y_T32.npy
    ├── cic17_W32_S16_test_X_T32.npy
    └── cic17_W32_S16_test_y_T32.npy
```

 4.3 Run the preprocessing pipeline

Run the four scripts **in strict order** (the numbering reflects the dependency chain):

```bash
python 02_clean_cic2017.py          # 1. clean + 7:2:1 train/val/test split (20 blocks)
python 04_feature_cic2017.py        # 2. sliding-window (W=32, S=16) features -> 150-dim
python 10_standardize_data.py       # 3. MinMaxScaler standardization (fit on train only)
python 11_convert_to_tft_tensor.py  # 4. convert to [N, 32, 150] tensors (.npy)
```

After step 4, the six `.npy` files under `tensor_T32/` are ready for training/evaluation.

 5. Running the Models

All model scripts auto-select CUDA (`torch.cuda.is_available()`). On first run they train
from scratch and save the best checkpoint (by validation F1); on later runs they load the
checkpoint, then evaluate on the test set and write results.

 5.1 LightTFT (baseline)

```bash
python LightTFT.py
```

Outputs go to `…/lighttft_linear_output/`: `lighttft_linear_metrics.json`,
`results_linear.npz`, `plots/`, and checkpoint `…/saved_models/baseline_linear_attn.pth`.

 5.2 WaveLightTFT (proposed)

```bash
python wavelighttft.py
```

The wavelet branch is computed on the fly and cached under `wavelet_cache_dir`. Outputs go
to `…/wavelighttft_output/`: `wavelighttft_metrics.json`, `results.npz`, `plots/`, and
checkpoint `…/saved_models/best_wavelighttft_dynamic.pth`.

> `LightTFT.py` and `wavelighttft.py` both `import tft_core`, so all three files must sit
> in the same directory (they do, at the repo root). Run these two scripts from the repo
> root.

 5.3 Wavelet hyper-parameter search (optional, `03小波/`)

```bash
PYTHONPATH=. python 03小波/002-Copy1.py   # wavelet basis (db4/db8/sym4/coif3/bior3.3)
PYTHONPATH=. python 03小波/002-Copy2.py   # decomposition level (2/3/4/[2,3])
PYTHONPATH=. python 03小波/002-Copy3.py   # fusion strategy (concat/attention/add/gate)
PYTHONPATH=. python 03小波/002-Copy4.py   # fusion position (early/middle)
PYTHONPATH=. python 03小波/002-Copy5.py   # weighting strategy (fixed/dynamic)
PYTHONPATH=. python 03小波/002-Copy6.py   # native Light-TFT baseline
PYTHONPATH=. python 03小波/002-Copy7.py   # robustness stress test
```

These scripts `import tft_core as core`, but `tft_core.py` lives at the repo root, so run
them **from the repo root with `PYTHONPATH=.`** (or copy `tft_core.py` into `03小波/`).

 5.4 Ablation & distillation (optional, `02轻量化/`)

`01_TFT_Medium_Test.py`, `02_TFT_Optuna_Optimization.py`, `04_Lightweight_Evolution_Final.py`
and `06_generate_teacher_logits.py` import `tft_binary.py`, which is **not included** in
this repository. Provide your own `tft_binary.py` (defining class `TFTBinary`) before
running these. `07_distill_std_run.py` is self-contained.

 6. Output Artifacts

Each model run produces:

- JSON metrics — Accuracy, F1, Precision, Recall, AUPRC, FNR, confusion matrix,
  *LOPs, Params, and per-sample inference time.
- NPZ — probabilities, labels, extracted features, inference time.
- Plots — loss curve, confusion matrix, t-SNE.
- Checkpoint — best `.pth` state dict.

 7. Notes & Known Issues

- **FLOPs / Params correctness**: `thop` has no built-in rule for `nn.MultiheadAttention`
  and would otherwise report it as 0, which made WaveLightTFT appear *smaller* than
  LightTFT. `wavelighttft.py` registers a custom counting hook
  (`count_multihead_attention`) and computes Params via `sum(p.numel())`, so the reported
  numbers are correct. Expected ordering:
  WaveLightTFT (≈ 3.08M FLOPs / ≈ 92.4K params) > LightTFT (≈ 2.64M / ≈ 82.7K).
- **Paths**: scripts hardcode `/root/autodl-tmp/graduate-thesis/…`. Adapt them to your
  environment (Section 4.2).
- **Encoding**: scripts print emoji and CJK text. On Windows set `PYTHONUTF8=1`
  (or `PYTHONIOENCODING=utf-8`) to avoid GBK console errors.
- **Missing module**: `tft_binary.py` is referenced by `02轻量化/` but not shipped —
  supply it yourself if you need those experiments.
- **Reproducibility**: training shuffles data each epoch. Set a manual seed
  (`torch.manual_seed`) if deterministic runs are required.

# SparseCNN-MAE

Fully convolutional Video Masked Autoencoder (Video-MAE) with a sparse 3D convolution encoder (spconv v2 first, dense fallback), designed for:
- self-supervised pretraining on raw VoxCeleb2 videos
- fine-tuning on DISFA for per-frame AU detection (`B x num_aus x T`)

Default settings are memory-safe for a single RTX 3060 12GB (AMP + small batch + grad accumulation).

## Highlights
- No ViT/Transformer backbone; fully convolutional 3D Conv design.
- 3D patch embedding (`Conv3d`, stride = kernel = patch size).
- Patch-level random masking over `(t,h,w)` grid.
- Sparse encoder using `spconv.SparseConvTensor` on visible patches only.
- Decoder: sparse->dense boundary at bottleneck, then dense `ConvTranspose3d` reconstruction.
- Raw video decoding from files (no frame pre-extraction required) with backend fallback:
  1. `torchvision.io.VideoReader`
  2. `pyav`
  3. `decord`
- Optional face alignment (`off`, `mtcnn`, `insightface`) with clip cache.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── smoke_test.py
├── configs/
│   ├── pretrain_voxceleb2.yaml
│   └── finetune_disfa.yaml
├── manifests/
├── scripts/
│   ├── make_voxceleb2_manifest.py
│   └── make_disfa_manifest.py
├── preprocess/
│   ├── __init__.py
│   ├── face_align.py
│   └── align_dataset.py
├── train/
│   ├── __init__.py
│   ├── pretrain.py
│   ├── finetune.py
│   └── evaluate.py
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── augment.py
    │   ├── disfa.py
    │   ├── video_reader.py
    │   └── voxceleb2.py
    ├── models/
    │   ├── __init__.py
    │   ├── au_head.py
    │   ├── decoder.py
    │   ├── encoder.py
    │   ├── mae.py
    │   ├── masking.py
    │   ├── patch_embed.py
    │   └── sparse_blocks.py
    ├── train/
    │   ├── __init__.py
    │   ├── evaluate.py
    │   ├── finetune.py
    │   └── pretrain.py
    └── utils/
        ├── __init__.py
        ├── checkpoint.py
        ├── config.py
        ├── logging.py
        ├── metrics.py
        ├── seed.py
        └── version.py
```

## Install

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Install spconv v2 (recommended)
Pick the wheel matching your CUDA/PyTorch stack.

Example CUDA 12.1:

```bash
pip install spconv-cu121>=2.3.8
```

If spconv is missing/incompatible, the code falls back to a dense Conv3d encoder and prints a warning.

### PyTorch 2.10 note
This repo targets PyTorch 2.10. If `2.10.x` is unavailable in your environment, code still runs on PyTorch `2.x` with a runtime warning.

## Build Dataset Manifests

### VoxCeleb2

```bash
python scripts/make_voxceleb2_manifest.py \
  --root /path/to/voxceleb2 \
  --out_dir manifests \
  --val_ratio 0.02
```

Outputs:
- `manifests/voxceleb2_train.csv`
- `manifests/voxceleb2_val.csv`

### DISFA

```bash
python scripts/make_disfa_manifest.py \
  --root /path/to/disfa \
  --videos_root /path/to/disfa/Videos \
  --annotations_root /path/to/disfa/ActionUnit_Labels \
  --out_csv manifests/disfa_manifest.csv
```

Manifest supports either `video_path` or `frame_dir` per row.

## Optional Face Alignment

Default is `off`.

### On-the-fly
Set in config or CLI override:

```bash
... data.align_mode=mtcnn data.align_cache_dir=/tmp/disfa_align_cache
```

### Offline

```bash
python -m preprocess.align_dataset \
  --input_manifest manifests/disfa_manifest.csv \
  --output_dir /tmp/disfa_aligned \
  --output_manifest manifests/disfa_manifest_aligned.csv \
  --mode mtcnn \
  --clip_len 32 \
  --stride 1
```

Pros:
- stable crops
- lower training-time CPU overhead

Cons:
- extra storage
- less augmentation diversity if cache is fixed

## Pretraining (VoxCeleb2)

### Smoke (50 optimizer updates)

```bash
python -m train.pretrain \
  --config configs/pretrain_voxceleb2.yaml \
  --smoke \
  data.root=/path/to/voxceleb2 \
  data.train_manifest=manifests/voxceleb2_train.csv \
  data.val_manifest=manifests/voxceleb2_val.csv
```

### Full

```bash
python -m train.pretrain \
  --config configs/pretrain_voxceleb2.yaml \
  data.root=/path/to/voxceleb2 \
  data.train_manifest=manifests/voxceleb2_train.csv \
  data.val_manifest=manifests/voxceleb2_val.csv
```

TensorBoard:

```bash
tensorboard --logdir outputs/pretrain_voxceleb2/tb
```

## Fine-tuning (DISFA AU detection)

### Smoke (50 optimizer updates)

```bash
python -m train.finetune \
  --config configs/finetune_disfa.yaml \
  --smoke \
  --pretrained outputs/pretrain_voxceleb2/checkpoints/best.ckpt \
  data.root=/path/to/disfa \
  data.manifest=manifests/disfa_manifest.csv
```

### Full

```bash
python -m train.finetune \
  --config configs/finetune_disfa.yaml \
  --pretrained outputs/pretrain_voxceleb2/checkpoints/best.ckpt \
  data.root=/path/to/disfa \
  data.manifest=manifests/disfa_manifest.csv
```

TensorBoard:

```bash
tensorboard --logdir outputs/finetune_disfa/tb
```

## Evaluation

```bash
python -m train.evaluate \
  --config configs/finetune_disfa.yaml \
  ckpt=outputs/finetune_disfa/checkpoints/best.ckpt \
  --split val \
  data.manifest=manifests/disfa_manifest.csv
```

Outputs JSON with:
- `loss`
- `per_au_f1`, `mean_f1`
- optional `per_au_auc`, `mean_auc`

## Smoke Test

```bash
python smoke_test.py
```

Checks:
- model build
- forward on `[2,3,32,112,112]`
- masked reconstruction loss path
- AMP backward

## Memory Defaults (RTX 3060 12GB)

Default configs use:
- `batch_size: 2`
- `accum_steps: 8`
- AMP on
- grad clipping

Increase `batch_size` only after verifying memory headroom.

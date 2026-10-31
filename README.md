# Self-Supervised Representation Learning — ITCS 6169/8169 Assignment 2

16-class scene recognition from 2,400 training images, with **no external pretraining**: every network
starts from random weights and sees only the assignment data. The representation is learned with
SimCLR on the training images, then fine-tuned with MixUp/CutMix and label smoothing.

**Best model:** ResNet-18 · SimCLR pretraining → fine-tuning (backbone lr 0.01) + MixUp/CutMix + label smoothing
**Test accuracy: 80.25%** (80.75% with horizontal-flip TTA) · val 84.2% · checkpoint: [`checkpoints/best_model.pt`](checkpoints/best_model.pt)

## Results

Val = 480 images held out from the training set (fixed split, seed 0). Test = 400 images, evaluated once
after all model choices were made on val.

| Method | Labels | Val | Test |
|---|---:|---:|---:|
| Supervised baseline | 1,920 | 78.5 | 76.0 |
| Supervised + SimCLR augmentation | 1,920 | 76.9 | 76.0 |
| SimCLR linear probe | 1,920 | 72.1 | 72.5 |
| SimCLR + fine-tune | 1,920 | 79.2 | 75.5 |
| SimCLR + fine-tune (backbone lr 0.01) | 1,920 | 78.3 | 75.8 |
| Joint CE + SimCLR | 1,920 | 80.6 | 72.5 |
| Supervised + MixUp/CutMix + LS | 1,920 | 82.7 | 82.0 |
| **SimCLR + fine-tune + MixUp/CutMix + LS (best)** | 1,920 | **84.2** | **80.3** |
| Supervised | 192 (10%) | 51.5 | – |
| SimCLR linear probe | 192 (10%) | 62.3 | – |
| SimCLR + fine-tune (backbone lr 0.01) | 192 (10%) | 65.8 | – |

SimCLR pretraining alone reaches 71.0% kNN accuracy (k = 20) on val without using any labels.
Per-run logs and metrics are in `runs/<name>/`; figures are in `figures/`.

## Repository contents

| Path | Contents |
|---|---|
| `experiments.ipynb` | Runs every experiment in order and shows logs, curves and result tables |
| `pretrain_ssl.py` | SimCLR pretraining (NT-Xent loss, projection head, kNN monitor) |
| `train.py` | Supervised training, fine-tuning, linear probing and joint CE + SimCLR training |
| `eval.py` | Inference / evaluation of a checkpoint on an image folder |
| `models.py` | ResNet encoder, projection head, classifier |
| `losses.py` | NT-Xent (SimCLR) loss |
| `augment.py` | Batched GPU augmentations (random resized crop, flip, colour jitter, grayscale) |
| `engine.py` | Evaluation, kNN accuracy, learning-rate schedule |
| `common.py` | Config loading, data loading and caching, train/val split |
| `plots.py` | Plotting and result-table helpers used by the notebook |
| `configs/` | Hyperparameters for each experiment (`best.yaml` is the final model) |
| `checkpoints/best_model.pt` | Trained weights of the best model |
| `runs/` | `config.json`, `log.csv`, `metrics.json`, `test.json` per run (weights not included) |
| `figures/` | Figures produced by the notebook |
| `experiments.py`, `tools/build_notebook.py` | Source of the notebook and the script that generates it |

## Setup

Tested with Python 3.11, PyTorch 2.6.0 + CUDA 12.4 on an NVIDIA RTX 4060 Laptop GPU (8 GB).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Data

Place the dataset as one folder per class:

```
data/
  train/<class_name>/*.jpg     # 2,400 images, 16 classes
  test/<class_name>/*.jpg      # 400 images
```

On first use the images are resized (short side 144 px, centre crop) and cached to `data/cache_144.pt`.
Use `--set data_root=<path>` (or `DATA_ROOT` in the notebook) if the data is elsewhere.

## Evaluate the best model

```bash
python eval.py --ckpt checkpoints/best_model.pt --data data/test
```

Prints overall and per-class test accuracy. Add `--tta` to average predictions with a horizontally
flipped copy of each image.

## Reproduce the best model from scratch

```bash
python pretrain_ssl.py --config configs/simclr.yaml --name simclr
python train.py --config configs/best.yaml --name best
python eval.py --run runs/best --data data/test
```

1. SimCLR pretraining on the 1,920 training-split images, labels unused (500 epochs, ~40 min on the GPU above).
   Writes `runs/simclr/encoder.pt`.
2. Fine-tuning from that encoder (3,000 steps, ~10–15 min). Writes `runs/best/model.pt` and val metrics.
3. Test-set evaluation. Writes `runs/best/test.json`.

Runs use seed 0, but GPU kernels are not bit-exact, so a re-run can differ by about ±1–2 points.

## Reproduce all experiments

Open `experiments.ipynb`, set `DATA_ROOT` in the first cell, and run all cells. Each cell launches a
script and streams its log. Runs that already have `runs/<name>/metrics.json` are skipped, so the
notebook can be re-run safely. Any config value can be overridden from the command line, for example:

```bash
python train.py --config configs/finetune.yaml --name my_run --set init=runs/simclr/encoder.pt lr=0.05 label_frac=0.1
```

## Method summary

- **Backbone:** ResNet-18 (torchvision, `weights=None`), 128×128 input.
- **SimCLR:** two augmented views per image (random resized crop 0.2–1, flip, colour jitter, random grayscale);
  2-layer MLP projection head (512→512→128); NT-Xent loss with temperature 0.2 and batch size 256;
  SGD lr 0.2, cosine schedule, 500 epochs. The projection head is discarded after pretraining.
- **Fine-tuning (best model):** SGD + Nesterov, lr 0.1 for the classifier head and 0.01 for the pretrained encoder,
  cosine schedule, weight decay 5e-4, batch 128, 3,000 steps; random resized crop 0.35–1, flip, mild colour jitter;
  MixUp (α = 0.2) / CutMix (α = 1.0), one chosen per batch; label smoothing 0.1.
- **Data split:** 1,920 train / 480 val from the provided training set (`random_split`, seed 0).
  The final checkpoint of each run is used; no early stopping on val.

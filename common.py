"""Config handling, data loading (cached uint8 tensors), and small utilities."""
import argparse, copy, json, random, time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm', '.tif', '.tiff', '.webp'}


# ----------------------------------------------------------------------------- config

DEFAULTS = dict(
    # data
    data_root='data',          # contains train/<class>/ and test/<class>/
    runs_dir='runs',
    store_size=144,            # images cached as SxS (short side resize + center crop)
    img_size=128,              # network input resolution
    val_frac=0.20,
    seed=0,
    label_frac=1.0,            # fraction of labelled train images used (stratified)
    # model
    arch='resnet18',
    small_stem=False,          # True: drop the stem max-pool (2x spatial res, ~4x compute)
    init=None,                 # path to an SSL encoder.pt to initialise from
    freeze_encoder=False,      # True = linear probe
    # optimisation (supervised / fine-tune / joint)
    steps=3000,
    batch_size=128,
    optimizer='sgd',
    lr=0.1,
    encoder_lr_mult=1.0,       # backbone lr = lr * mult (head uses lr)
    weight_decay=5e-4,
    warmup_steps=150,
    label_smoothing=0.0,
    mixup=0.0,                 # Beta(a, a) alpha, 0 = off
    cutmix=0.0,                # Beta(a, a) alpha, 0 = off
    eval_every=150,
    aug='sup',                 # preset name or dict, see augment.PRESETS
    # joint SSL + CE
    ssl_weight=0.0,            # lambda for the SimCLR term; 0 = pure supervised
    ssl_aug='ssl',
    temperature=0.2,
    proj_dim=128,
    # SSL pretraining (pretrain_ssl.py)
    epochs=500,
    knn_every=25,
    knn_k=20,
    ssl_data='train',          # 'train' (1,920 split) or 'trainval' (all 2,400 train images)
    amp=True,
)


def _parse_value(v):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def _assign(cfg, key, value):
    from augment import resolve_aug
    parts = key.split('.')
    d = cfg
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value
    for k in ('aug', 'ssl_aug'):
        cfg[k] = resolve_aug(cfg[k])


def get_config(description=''):
    """defaults <- --config yaml <- --set key=value overrides."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--config', type=str, default=None)
    ap.add_argument('--name', type=str, default=None, help='run name (folder under runs_dir)')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                    help='override config values, e.g. --set lr=0.05 aug.scale_min=0.2')
    ap.add_argument('--force', action='store_true', help='re-run even if metrics.json exists')
    args = ap.parse_args()

    cfg = copy.deepcopy(DEFAULTS)
    from augment import resolve_aug
    for k in ('aug', 'ssl_aug'):
        cfg[k] = resolve_aug(cfg[k])
    if args.config:
        with open(args.config) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                _assign(cfg, k, v)
    for kv in args.set:
        k, v = kv.split('=', 1)
        _assign(cfg, k, _parse_value(v))
    cfg['name'] = args.name or (Path(args.config).stem if args.config else 'run')
    cfg['force'] = args.force
    return cfg


def run_dir(cfg):
    d = Path(cfg['runs_dir']) / cfg['name']
    d.mkdir(parents=True, exist_ok=True)
    return d


def already_done(cfg):
    done = (Path(cfg['runs_dir']) / cfg['name'] / 'metrics.json').exists()
    if done and not cfg['force']:
        print(f"[skip] {cfg['name']} already has metrics.json (use --force to re-run)")
        return True
    return False


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Logger:
    """Appends one CSV row per eval and echoes it to stdout (visible in the notebook)."""
    def __init__(self, path):
        self.path, self.header, self.t0 = Path(path), None, time.time()
        self.path.unlink(missing_ok=True)

    def log(self, **row):
        row = {'time_s': round(time.time() - self.t0, 1), **row}
        if self.header is None:
            self.header = list(row)
            self.path.write_text(','.join(self.header) + '\n')
        with open(self.path, 'a') as f:
            f.write(','.join(str(row.get(k, '')) for k in self.header) + '\n')
        print('  '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                        for k, v in row.items()), flush=True)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


# ----------------------------------------------------------------------------- data

def _scan(root):
    """Same ordering as torchvision ImageFolder: sorted classes, sorted files."""
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    items = []
    for ci, c in enumerate(classes):
        for p in sorted((root / c).rglob('*')):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, ci))
    return classes, items


def _load_image(path, size):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    img = img.crop((l, t, l + size, t + size))
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


def load_folder(root, size):
    classes, items = _scan(root)
    x = torch.stack([_load_image(p, size) for p, _ in items])
    y = torch.tensor([c for _, c in items])
    return x, y, classes


def load_data(cfg):
    """Returns uint8 image tensors and labels for train / val / test.

    Train/val split: random_split(range(n), seed 0), 20% val.
    """
    root = Path(cfg['data_root'])
    cache = root / f"cache_{cfg['store_size']}.pt"
    if cache.exists():
        d = torch.load(cache)
    else:
        print(f'Building image cache {cache} ...', flush=True)
        xtr, ytr, classes = load_folder(root / 'train', cfg['store_size'])
        xte, yte, classes_te = load_folder(root / 'test', cfg['store_size'])
        assert classes == classes_te, 'train/test class folders differ'
        d = dict(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte, classes=classes)
        torch.save(d, cache)

    n = len(d['y_train'])
    n_val = int(round(n * cfg['val_frac']))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g).tolist()
    tr_idx, va_idx = perm[:n - n_val], perm[n - n_val:]

    tr_idx_lab = tr_idx
    if cfg['label_frac'] < 1.0:
        rng = np.random.RandomState(cfg['seed'])
        y = d['y_train'][tr_idx].numpy()
        keep = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            k = max(1, int(round(len(idx) * cfg['label_frac'])))
            keep += rng.choice(idx, k, replace=False).tolist()
        tr_idx_lab = [tr_idx[i] for i in sorted(keep)]

    X, Y = d['x_train'], d['y_train']
    return dict(
        classes=d['classes'],
        x_train=X[tr_idx_lab], y_train=Y[tr_idx_lab],        # labelled training set
        x_train_all=X[tr_idx], y_train_all=Y[tr_idx],        # full 1,920 split (SSL / kNN bank)
        x_trainval=X, y_trainval=Y,
        x_val=X[va_idx], y_val=Y[va_idx],
        x_test=d['x_test'], y_test=d['y_test'],
    )

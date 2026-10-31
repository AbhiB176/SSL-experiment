# %% [markdown]
# # Assignment 2: Self-Supervised Representation Learning

# %% [markdown]
# ## 0. Setup

# %%
import subprocess, sys
from pathlib import Path

import torch

DATA_ROOT = Path('data')       # contains train/<class>/ and test/<class>/  (edit for your machine)
RUNS_DIR = Path('runs')
FIGS = Path('figures')

print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}'
      + (f' | {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ''))
assert (DATA_ROOT / 'train').exists(), f'put the dataset under {DATA_ROOT.resolve()}'


def run(script, config, name, *sets, force=False):
    """Run a training script with the same Python as this kernel and stream its output."""
    cmd = [sys.executable, '-u', script, '--config', f'configs/{config}.yaml', '--name', name,
           '--set', f'data_root={DATA_ROOT.as_posix()}', f'runs_dir={RUNS_DIR.as_posix()}', *sets]
    if force:
        cmd.append('--force')
    print('$', ' '.join(cmd[1:]), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    if p.wait() != 0:
        raise RuntimeError(f'{name} failed (exit code {p.returncode})')

# %% [markdown]
# ## 1. Data and augmentations

# %%
import matplotlib.pyplot as plt
import plots
from augment import PRESETS, augment, denormalize
from common import DEFAULTS, load_data

cfg = dict(DEFAULTS, data_root=str(DATA_ROOT))
data = load_data(cfg)
classes = data['classes']
print(f"{len(classes)} classes | train {len(data['y_train'])} | val {len(data['y_val'])} | test {len(data['y_test'])}")
print(classes)

torch.manual_seed(1)
idx = [0, 300, 700, 1100]
fig, axes = plt.subplots(len(idx), 7, figsize=(14, 2.1 * len(idx)))
for r, i in enumerate(idx):
    x = data['x_train'][i:i + 1].repeat(3, 1, 1, 1)
    views = [x[:1].float() / 255] + [denormalize(v) for v in augment(x, PRESETS['sup'], 128)] \
            + [denormalize(v) for v in augment(x, PRESETS['ssl'], 128)]
    for c, v in enumerate(views):
        axes[r, c].imshow(v.squeeze(0).permute(1, 2, 0)); axes[r, c].axis('off')
    axes[r, 0].set_title(classes[data['y_train'][i]], fontsize=9)
for c, t in enumerate(['original'] + ['sup'] * 3 + ['ssl'] * 3):
    if c: axes[0, c].set_title(t, fontsize=9)
plt.tight_layout(); FIGS.mkdir(exist_ok=True); plt.savefig(FIGS / 'augmentations.png', dpi=120); plt.show()

# %% [markdown]
# ## 2. Supervised baseline

# %%
run('train.py', 'supervised', 'sup_baseline')
plots.curves(['sup_baseline'], RUNS_DIR, save=FIGS / 'exp1_baseline.png')
plots.table(['sup_baseline'], RUNS_DIR)

# %% [markdown]
# ## 3. SimCLR pretraining

# %%
run('pretrain_ssl.py', 'simclr', 'simclr')
plots.ssl_curves(['simclr'], RUNS_DIR, save=FIGS / 'exp2_simclr_pretrain.png')

# %% [markdown]
# ### Linear probe and fine-tuning

# %%
ENC = (RUNS_DIR / 'simclr' / 'encoder.pt').as_posix()
run('train.py', 'probe', 'simclr_probe', f'init={ENC}')
run('train.py', 'finetune', 'simclr_ft', f'init={ENC}')
plots.curves(['sup_baseline', 'simclr_ft'], RUNS_DIR, labels=['supervised (random init)', 'SimCLR init + fine-tune'],
             save=FIGS / 'exp2_ft_vs_sup.png')
plots.table(['sup_baseline', 'simclr_probe', 'simclr_ft'], RUNS_DIR)

# %%
run('train.py', 'finetune', 'simclr_ft_lowlr', f'init={ENC}', 'encoder_lr_mult=0.1')
plots.curves(['sup_baseline', 'simclr_ft', 'simclr_ft_lowlr'], RUNS_DIR,
             labels=['supervised', 'SimCLR + FT (backbone lr 0.1)', 'SimCLR + FT (backbone lr 0.01)'],
             save=FIGS / 'exp2_ft_lr.png')
plots.table(['sup_baseline', 'simclr_probe', 'simclr_ft', 'simclr_ft_lowlr'], RUNS_DIR)

# %% [markdown]
# ## 4. Joint CE + SimCLR, strong-augmentation supervised control

# %%
run('train.py', 'supervised_strongaug', 'sup_strongaug')
run('train.py', 'joint', 'joint')
plots.curves(['sup_baseline', 'sup_strongaug', 'joint', 'simclr_ft'], RUNS_DIR, save=FIGS / 'exp3_joint.png')
plots.table(['sup_baseline', 'sup_strongaug', 'simclr_probe', 'simclr_ft', 'simclr_ft_lowlr', 'joint'], RUNS_DIR)

# %% [markdown]
# ## 5. Label efficiency

# %%
FRACS = [0.1]
for f in FRACS:
    run('train.py', 'supervised', f'sup_frac{f}', f'label_frac={f}')
    run('train.py', 'finetune', f'ft_frac{f}', f'label_frac={f}', f'init={ENC}', 'encoder_lr_mult=0.1')
    run('train.py', 'probe', f'probe_frac{f}', f'label_frac={f}', f'init={ENC}')

# %%
import shutil
for a, b in [('sup_baseline', 'sup_frac1.0'), ('simclr_ft_lowlr', 'ft_frac1.0'), ('simclr_probe', 'probe_frac1.0')]:
    if (RUNS_DIR / a / 'metrics.json').exists() and not (RUNS_DIR / b).exists():
        shutil.copytree(RUNS_DIR / a, RUNS_DIR / b, ignore=shutil.ignore_patterns('*.pt'))
plots.label_efficiency({'supervised (random init)': 'sup_frac{f}', 'SimCLR + fine-tune': 'ft_frac{f}',
                        'SimCLR linear probe': 'probe_frac{f}'}, FRACS + [1.0], RUNS_DIR,
                       save=FIGS / 'exp4_label_efficiency.png')

# %% [markdown]
# ## 6. Best recipe

# %%
REG = ['mixup=0.2', 'cutmix=1.0', 'label_smoothing=0.1']
run('train.py', 'finetune', 'best', f'init={ENC}', 'encoder_lr_mult=0.1', *REG)
plots.curves(['sup_baseline', 'simclr_ft_lowlr', 'best'], RUNS_DIR,
             labels=['supervised', 'SimCLR + FT (backbone lr 0.01)', 'SimCLR + FT + MixUp/CutMix + LS'],
             save=FIGS / 'best.png')
plots.table(['sup_baseline', 'simclr_ft_lowlr', 'best'], RUNS_DIR)

# %%
run('train.py', 'supervised', 'sup_reg', *REG)
plots.curves(['sup_baseline', 'sup_reg', 'best'], RUNS_DIR,
             labels=['supervised', 'supervised + MixUp/CutMix + LS', 'SimCLR + FT + MixUp/CutMix + LS'],
             save=FIGS / 'best_vs_sup_reg.png')
plots.table(['sup_baseline', 'sup_reg', 'simclr_ft_lowlr', 'best'], RUNS_DIR)

# %% [markdown]
# ## 7. Test set evaluation

# %%
FINAL = ['sup_baseline', 'sup_strongaug', 'simclr_probe', 'simclr_ft', 'simclr_ft_lowlr', 'joint', 'sup_reg', 'best']
cmd = [sys.executable, 'eval.py', '--data', (DATA_ROOT / 'test').as_posix(),
       '--run', *[(RUNS_DIR / n).as_posix() for n in FINAL if (RUNS_DIR / n / 'model.pt').exists()]]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout, r.stderr)
r.check_returncode()

# %% [markdown]
# ## 8. Summary

# %%
FINAL = ['sup_baseline', 'sup_strongaug', 'simclr_probe', 'simclr_ft', 'simclr_ft_lowlr', 'joint', 'sup_reg', 'best']
LABELS = ['Supervised baseline', 'Supervised + SSL aug', 'SimCLR linear probe', 'SimCLR + fine-tune',
          'SimCLR + fine-tune (backbone lr 0.01)', 'Joint CE + SimCLR', 'Supervised + MixUp/CutMix + LS',
          'Best recipe']
plots.table(FINAL, RUNS_DIR, labels=LABELS)

# %%
import importlib, plots
importlib.reload(plots)

plots.bars(FINAL, RUNS_DIR, labels=LABELS, save=FIGS / 'summary_val_test.png')
plots.curves(FINAL, RUNS_DIR, labels=LABELS, show_train=False, save=FIGS / 'summary_val_curves.png')

# %%
plots.per_class(FINAL, RUNS_DIR, labels=LABELS, split='test', save=FIGS / 'summary_per_class_test.png')

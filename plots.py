"""Notebook helpers: read run logs/metrics and draw the figures used in the report."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

RUNS = Path('runs')


def log(name, runs=RUNS):
    return pd.read_csv(Path(runs) / name / 'log.csv')


def metrics(name, runs=RUNS):
    p, t = Path(runs) / name / 'metrics.json', Path(runs) / name / 'test.json'
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    if t.exists():
        m.update(json.loads(t.read_text()))
    return m


def curves(names, runs=RUNS, labels=None, save=None, show_train=True):
    """Loss and accuracy curves for supervised-style runs. Dashed = clean train, solid = val."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    for i, n in enumerate(names):
        df, lab, c = log(n, runs), (labels or names)[i], f'C{i}'
        if show_train:
            a1.plot(df.step, df.train_loss_clean, '--', color=c, alpha=.6)
            a2.plot(df.step, df.train_acc_clean, '--', color=c, alpha=.6)
        a1.plot(df.step, df.val_loss, color=c, label=lab)
        a2.plot(df.step, df.val_acc, color=c, label=lab)
    sub = ' (dashed = train, solid = val)' if show_train else ' (val)'
    a1.set(xlabel='step', ylabel='loss' + sub)
    a2.set(xlabel='step', ylabel='accuracy' + sub, ylim=(0, 1.02) if show_train else None)
    for ax in (a1, a2):
        handles = ax.get_legend_handles_labels()[0]
        if show_train:
            handles += [Line2D([], [], color='gray', ls='--', alpha=.6, label='train (clean, no aug)'),
                        Line2D([], [], color='gray', label='val')]
        ax.legend(handles=handles, fontsize=8)
    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True); plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


def ssl_curves(names, runs=RUNS, labels=None, save=None):
    """SimCLR loss and kNN val accuracy over pretraining epochs."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    for i, n in enumerate(names):
        df, lab = log(n, runs), (labels or names)[i]
        a1.plot(df.epoch, df.ssl_loss, label=lab)
        a2.plot(df.epoch, df.knn_val_acc, marker='o', ms=3, label=lab)
    a1.set(xlabel='epoch', ylabel='NT-Xent loss'); a2.set(xlabel='epoch', ylabel='kNN val accuracy (k=20)')
    a1.legend(); a2.legend(); plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True); plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


def table(names, runs=RUNS, labels=None):
    rows = []
    for i, n in enumerate(names):
        m = metrics(n, runs)
        if m is None:
            continue
        rows.append({'method': (labels or names)[i], 'labels': m.get('n_labelled'),
                     'val': m.get('val_acc'), 'val+TTA': m.get('val_acc_tta'),
                     'test': m.get('test_acc'), 'test+TTA': m.get('test_acc_tta'),
                     'train(clean)': m.get('train_acc_clean'), 'gap': m.get('gap'),
                     'kNN val': m.get('knn_val_acc'), 'min': m.get('minutes')})
    df = pd.DataFrame(rows).set_index('method')
    pct = [c for c in df.columns if c not in ('labels', 'min')]
    return df.style.format(lambda v: f'{v * 100:.1f}', subset=pct, na_rep='-')\
        .format(na_rep='-', precision=1, subset=['min']).format('{:.0f}', na_rep='-', subset=['labels'])


def label_efficiency(groups, fracs, runs=RUNS, metric='val_acc', save=None):
    """groups: {'Supervised': 'sup_frac{f}', 'SimCLR + FT': 'ft_frac{f}'} name templates."""
    plt.figure(figsize=(6, 4))
    for lab, tmpl in groups.items():
        ys = [(metrics(tmpl.format(f=f), runs) or {}).get(metric) for f in fracs]
        plt.plot([f * 100 for f in fracs], [y * 100 if y else None for y in ys], marker='o', label=lab)
    plt.xlabel('% of training labels used'); plt.ylabel(f'{metric} (%)'); plt.xscale('log')
    plt.xticks([f * 100 for f in fracs], [f'{f * 100:g}' for f in fracs]); plt.grid(alpha=.3); plt.legend()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True); plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


def bars(names, runs=RUNS, labels=None, save=None):
    """Final val and test accuracy per model."""
    labels = labels or names
    ms = [metrics(n, runs) or {} for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4))
    for off, key, lab in [(-0.2, 'val_acc', 'val'), (0.2, 'test_acc', 'test')]:
        ys = [m.get(key, np.nan) * 100 for m in ms]
        b = ax.bar(x + off, ys, 0.4, label=lab)
        ax.bar_label(b, fmt='%.1f', fontsize=8, padding=2)
    ax.set_xticks(x, labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('accuracy (%)'); ax.set_ylim(50, 100); ax.legend(); ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True); plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


def per_class(names, runs=RUNS, labels=None, split='test', save=None):
    """Per-class accuracy heatmap (models x classes) from test.json / metrics.json."""
    labels = labels or names
    key = f'per_class_{split}'
    rows = {lab: (metrics(n, runs) or {}).get(key) for n, lab in zip(names, labels)}
    df = pd.DataFrame({k: v for k, v in rows.items() if v}).T * 100
    fig, ax = plt.subplots(figsize=(13, 0.45 * len(df) + 1.5))
    im = ax.imshow(df.values, cmap='RdYlGn', vmin=30, vmax=100, aspect='auto')
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(df)), df.index, fontsize=9)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f'{df.values[i, j]:.0f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label=f'{split} accuracy (%)', fraction=0.025)
    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True); plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
    return df.round(1)

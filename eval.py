"""Evaluate trained checkpoints on an image folder (default: the test set).

    python eval.py --ckpt checkpoints/best_model.pt --data data/test [--tta]
    python eval.py --run runs/sup_baseline runs/simclr_ft --data data/test   # writes runs/<name>/test.json
"""
import argparse
import json
from pathlib import Path

import torch

from common import get_device, load_folder
from engine import evaluate
from models import Classifier


def load_model(path, dev):
    ck = torch.load(path, map_location='cpu')
    model = Classifier(ck['arch'], len(ck['classes']), ck['small_stem'], head_bn=ck.get('head_bn', False))
    model.load_state_dict({k: v for k, v in ck['model'].items() if not k.startswith('proj.')})
    return model.to(dev), ck


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--ckpt', help='path to a model.pt checkpoint')
    g.add_argument('--run', nargs='+', help='run folders; results are written to <run>/test.json')
    ap.add_argument('--data', default='data/test', help='folder with one sub-folder per class')
    ap.add_argument('--tta', action='store_true', help='average predictions with horizontal flip')
    args = ap.parse_args()

    dev = get_device()
    targets = [Path(args.ckpt)] if args.ckpt else [Path(r) / 'model.pt' for r in args.run]
    cache = {}
    for path in targets:
        model, ck = load_model(path, dev)
        if ck['store_size'] not in cache:
            cache[ck['store_size']] = load_folder(args.data, ck['store_size'])
        x, y, classes = cache[ck['store_size']]
        assert classes == ck['classes'], f'class folders {classes} != checkpoint classes {ck["classes"]}'
        _, acc, pred = evaluate(model, x, y, ck['img_size'], dev)
        _, acc_tta, pred_tta = evaluate(model, x, y, ck['img_size'], dev, tta=True)
        p = pred_tta if args.tta else pred
        per_class = {c: (p[y == i] == i).float().mean().item() for i, c in enumerate(classes)}
        print(f'{path}: {len(y)} images | accuracy {acc * 100:.2f}% | flip TTA {acc_tta * 100:.2f}%')
        if args.ckpt:
            for c, a in per_class.items():
                print(f'  {c:<14} {a * 100:6.1f}%')
        else:
            (path.parent / 'test.json').write_text(json.dumps(
                {'test_acc': acc, 'test_acc_tta': acc_tta, 'per_class_test': per_class}, indent=2))


if __name__ == '__main__':
    main()

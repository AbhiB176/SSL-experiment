"""Supervised training, SSL fine-tuning, linear probing, and joint CE + SimCLR training.

    # supervised baseline (random init)
    python train.py --config configs/supervised.yaml
    # fine-tune / linear-probe an SSL encoder
    python train.py --config configs/finetune.yaml --set init=runs/simclr/encoder.pt
    # joint objective: CE + lambda * NT-Xent
    python train.py --config configs/joint.yaml
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from augment import augment
from common import Logger, already_done, get_config, get_device, load_data, run_dir, save_json, set_seed
from engine import cosine_lr, evaluate
from losses import nt_xent
from models import Classifier, load_encoder_weights


def mix_batch(x, y, cfg, num_classes):
    """MixUp / CutMix with soft targets. One of the two is chosen per batch if both are on."""
    y1 = F.one_hot(y, num_classes).float()
    choices = [m for m in ('mixup', 'cutmix') if cfg[m] > 0]
    if not choices:
        return x, y1
    kind = choices[np.random.randint(len(choices))]
    lam = float(np.random.beta(cfg[kind], cfg[kind]))
    perm = torch.randperm(len(x), device=x.device)
    if kind == 'mixup':
        x = lam * x + (1 - lam) * x[perm]
    else:
        H, W = x.shape[2:]
        rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y0, y1_, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, H), max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone()
        x[:, :, y0:y1_, x0:x1] = x[perm, :, y0:y1_, x0:x1]
        lam = 1 - (y1_ - y0) * (x1 - x0) / (H * W)
    return x, lam * y1 + (1 - lam) * y1[perm]


def soft_ce(logits, target, smoothing):
    if smoothing > 0:
        target = target * (1 - smoothing) + smoothing / target.shape[1]
    return -(target * F.log_softmax(logits.float(), 1)).sum(1).mean()


def main():
    cfg = get_config('Supervised / fine-tune / probe / joint training')
    if already_done(cfg):
        return
    out = run_dir(cfg)
    save_json(cfg, out / 'config.json')
    set_seed(cfg['seed'])
    dev = get_device()
    data = load_data(cfg)
    C = len(data['classes'])
    xtr, ytr = data['x_train'].to(dev), data['y_train'].to(dev)
    N, bs = len(xtr), min(cfg['batch_size'], len(xtr))
    joint = cfg['ssl_weight'] > 0

    model = Classifier(cfg['arch'], C, cfg['small_stem'], cfg['proj_dim'] if joint else None,
                       head_bn=cfg['freeze_encoder'])
    if cfg['init']:
        load_encoder_weights(model, cfg['init'])
    if cfg['freeze_encoder']:
        for p in model.encoder.parameters():
            p.requires_grad = False
    model = model.to(dev).to(memory_format=torch.channels_last)

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    other = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    groups = [{'params': other, 'lr_mult': 1.0}]
    if enc_params:
        groups.append({'params': enc_params, 'lr_mult': cfg['encoder_lr_mult']})
    if cfg['optimizer'] == 'sgd':
        opt = torch.optim.SGD(groups, lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay'], nesterov=True)
    else:
        opt = torch.optim.AdamW(groups, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    use_amp = cfg['amp'] and dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    mode = ('probe' if cfg['freeze_encoder'] else 'finetune' if cfg['init'] else 'scratch') + ('+joint' if joint else '')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{cfg['name']} | {mode} | {cfg['arch']} | {N} labelled imgs | {cfg['steps']} steps "
          f"bs {bs} | trainable {n_params / 1e6:.2f}M | aug={cfg['ssl_aug'] if joint else cfg['aug']}", flush=True)

    log = Logger(out / 'log.csv')
    perm, ptr = torch.randperm(N, device=dev), 0
    run_ce, run_ssl, run_acc, n_run = 0.0, 0.0, 0.0, 0
    for step in range(cfg['steps']):
        lr = cosine_lr(step, cfg['steps'], cfg['lr'], cfg['warmup_steps'])
        for g in opt.param_groups:
            g['lr'] = lr * g['lr_mult']
        if ptr + bs > N:
            perm, ptr = torch.randperm(N, device=dev), 0
        idx = perm[ptr:ptr + bs]; ptr += bs
        xb, yb = xtr[idx], ytr[idx]

        model.train()
        if cfg['freeze_encoder']:
            model.encoder.eval()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            if joint:
                v1 = augment(xb, cfg['ssl_aug'], cfg['img_size'])
                v2 = augment(xb, cfg['ssl_aug'], cfg['img_size'])
                logits, h = model(torch.cat([v1, v2]).contiguous(memory_format=torch.channels_last), return_feat=True)
                target = F.one_hot(torch.cat([yb, yb]), C).float()
                ce = soft_ce(logits, target, cfg['label_smoothing'])
                z = model.proj(h)
                ssl = nt_xent(z[:bs], z[bs:], cfg['temperature'])
                loss = ce + cfg['ssl_weight'] * ssl
                acc_y = torch.cat([yb, yb])
            else:
                x = augment(xb, cfg['aug'], cfg['img_size'])
                x, target = mix_batch(x, yb, cfg, C)
                logits = model(x.contiguous(memory_format=torch.channels_last))
                ce = loss = soft_ce(logits, target, cfg['label_smoothing'])
                ssl = torch.zeros(())
                acc_y = yb
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        run_ce += ce.item(); run_ssl += ssl.item(); n_run += 1
        run_acc += (logits.argmax(1) == acc_y).float().mean().item()

        if (step + 1) % cfg['eval_every'] == 0 or step + 1 == cfg['steps']:
            vl, va, _ = evaluate(model, data['x_val'], data['y_val'], cfg['img_size'], dev)
            tl, ta, _ = evaluate(model, data['x_train'], data['y_train'], cfg['img_size'], dev)
            row = dict(step=step + 1, lr=lr, train_loss=run_ce / n_run, train_acc_aug=run_acc / n_run,
                       train_loss_clean=tl, train_acc_clean=ta, val_loss=vl, val_acc=va)
            if joint:
                row['ssl_loss'] = run_ssl / n_run
            log.log(**row)
            run_ce, run_ssl, run_acc, n_run = 0.0, 0.0, 0.0, 0

    vl, va, pred = evaluate(model, data['x_val'], data['y_val'], cfg['img_size'], dev)
    _, va_tta, _ = evaluate(model, data['x_val'], data['y_val'], cfg['img_size'], dev, tta=True)
    yv = data['y_val']
    per_class = {c: (pred[yv == i] == i).float().mean().item() for i, c in enumerate(data['classes'])}
    torch.save({'model': model.state_dict(), 'arch': cfg['arch'], 'small_stem': cfg['small_stem'], 'head_bn': cfg['freeze_encoder'],
                'img_size': cfg['img_size'], 'store_size': cfg['store_size'],
                'classes': data['classes']}, out / 'model.pt')
    metrics = dict(name=cfg['name'], type=mode, label_frac=cfg['label_frac'], n_labelled=N,
                   val_acc=va, val_loss=vl, train_acc_clean=ta, gap=ta - va,
                   val_acc_tta=va_tta, per_class_val=per_class,
                   minutes=round((time.time() - log.t0) / 60, 1))
    save_json(metrics, out / 'metrics.json')
    print(f"Done. val {va:.4f} | val+flipTTA {va_tta:.4f} | train-val gap {ta - va:.4f}", flush=True)


if __name__ == '__main__':
    main()

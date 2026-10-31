"""SimCLR pretraining on the training images (labels unused).

    python pretrain_ssl.py --config configs/simclr.yaml [--set epochs=200 ssl_aug.jitter=0]

Writes runs/<name>/{log.csv, encoder.pt, metrics.json}. kNN val accuracy is logged
every `knn_every` epochs.
"""
import time

import torch

from augment import augment
from common import Logger, already_done, get_config, get_device, load_data, run_dir, save_json, set_seed
from engine import cosine_lr, knn_accuracy
from losses import nt_xent
from models import ProjectionHead, make_encoder


def main():
    cfg = get_config('SimCLR pretraining')
    if already_done(cfg):
        return
    out = run_dir(cfg)
    save_json(cfg, out / 'config.json')
    set_seed(cfg['seed'])
    dev = get_device()
    data = load_data(cfg)

    x_ssl = data['x_trainval'] if cfg['ssl_data'] == 'trainval' else data['x_train_all']
    x_ssl = x_ssl.to(dev)
    N, bs = len(x_ssl), cfg['batch_size']
    steps_per_epoch = N // bs
    total = cfg['epochs'] * steps_per_epoch

    encoder, d = make_encoder(cfg['arch'], cfg['small_stem'])
    proj = ProjectionHead(d, cfg['proj_dim'])
    encoder, proj = encoder.to(dev).to(memory_format=torch.channels_last), proj.to(dev)
    params = list(encoder.parameters()) + list(proj.parameters())
    opt = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, weight_decay=cfg['weight_decay'])
    use_amp = cfg['amp'] and dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"SimCLR | {cfg['arch']} | {N} unlabeled images | bs {bs} | {cfg['epochs']} epochs "
          f"({total} steps) | T={cfg['temperature']} | aug={cfg['ssl_aug']}", flush=True)
    log = Logger(out / 'log.csv')
    step, knn = 0, None
    for epoch in range(1, cfg['epochs'] + 1):
        encoder.train(); proj.train()
        perm = torch.randperm(N, device=dev)
        tot = 0.0
        for i in range(steps_per_epoch):
            lr = cosine_lr(step, total, cfg['lr'], cfg['warmup_steps'])
            for g in opt.param_groups:
                g['lr'] = lr
            xb = x_ssl[perm[i * bs:(i + 1) * bs]]
            v1 = augment(xb, cfg['ssl_aug'], cfg['img_size']).contiguous(memory_format=torch.channels_last)
            v2 = augment(xb, cfg['ssl_aug'], cfg['img_size']).contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                z = proj(encoder(torch.cat([v1, v2])))
            loss = nt_xent(z[:len(xb)], z[len(xb):], cfg['temperature'])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); step += 1

        if epoch % cfg['knn_every'] == 0 or epoch == cfg['epochs'] or epoch == 1:
            knn = knn_accuracy(encoder, data['x_train_all'], data['y_train_all'],
                               data['x_val'], data['y_val'], cfg['img_size'], dev, k=cfg['knn_k'])
            log.log(epoch=epoch, step=step, lr=lr, ssl_loss=tot / steps_per_epoch, knn_val_acc=knn)
            torch.save({'encoder': encoder.state_dict(), 'proj': proj.state_dict(), 'epoch': epoch},
                       out / 'encoder.pt')

    save_json({'name': cfg['name'], 'type': 'ssl_pretrain', 'knn_val_acc': knn,
               'final_ssl_loss': tot / steps_per_epoch, 'epochs': cfg['epochs'],
               'minutes': round((time.time() - log.t0) / 60, 1)}, out / 'metrics.json')
    print(f'Done. kNN val acc {knn:.4f} -> {out / "encoder.pt"}')


if __name__ == '__main__':
    main()

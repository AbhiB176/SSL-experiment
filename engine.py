"""Evaluation helpers and the LR schedule, shared by all training scripts."""
import math

import torch
import torch.nn.functional as F

from augment import eval_transform


def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


@torch.inference_mode()
def extract(fn, x_uint8, img_size, device, bs=256, flip=False):
    """Apply fn (encoder or full model) to eval-transformed images in batches."""
    outs = []
    for i in range(0, len(x_uint8), bs):
        xb = eval_transform(x_uint8[i:i + bs].to(device), img_size)
        o = fn(xb)
        if flip:
            o = (o + fn(xb.flip(3))) / 2
        outs.append(o.float())
    return torch.cat(outs)


@torch.inference_mode()
def evaluate(model, x, y, img_size, device, tta=False):
    model.eval()
    logits = extract(lambda b: F.softmax(model(b), 1) if tta else model(b),
                     x, img_size, device, flip=tta)
    y = y.to(device)
    loss = F.nll_loss(logits.clamp_min(1e-8).log(), y).item() if tta else F.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc, logits.argmax(1).cpu()


@torch.inference_mode()
def knn_accuracy(encoder, x_bank, y_bank, x_query, y_query, img_size, device, k=20, t=0.07):
    """Weighted kNN (k, temperature t) on L2-normalised encoder features."""
    encoder.eval()
    fb = F.normalize(extract(encoder, x_bank, img_size, device), dim=1)
    fq = F.normalize(extract(encoder, x_query, img_size, device), dim=1)
    yb, yq = y_bank.to(device), y_query.to(device)
    C = int(yb.max()) + 1
    sim, idx = (fq @ fb.t()).topk(k, dim=1)
    w = (sim / t).exp()
    votes = torch.zeros(len(fq), C, device=device).scatter_add_(1, yb[idx], w)
    return (votes.argmax(1) == yq).float().mean().item()

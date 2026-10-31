"""Batched GPU augmentations on uint8 image tensors (independent parameters per image)."""
import math

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

MEAN, STD = 0.5, 0.25

PRESETS = {
    # no augmentation: full image resized to img_size (eval, linear probe)
    'none': dict(scale_min=1.0, scale_max=1.0, flip=False, jitter=0.0, hue=0.0,
                 jitter_p=0.0, gray_p=0.0),
    # supervised augmentation
    'sup':  dict(scale_min=0.35, scale_max=1.0, flip=True, jitter=0.2, hue=0.0,
                 jitter_p=0.8, gray_p=0.0),
    # SimCLR augmentation
    'ssl':  dict(scale_min=0.2, scale_max=1.0, flip=True, jitter=0.4, hue=0.1,
                 jitter_p=0.8, gray_p=0.2),
}


def resolve_aug(a):
    if isinstance(a, str):
        return dict(PRESETS[a])
    out = dict(PRESETS['none'])
    out.update(a)
    return out


def _gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def _uniform(n, lo, hi, device):
    return torch.empty(n, device=device).uniform_(lo, hi)


def random_resized_crop(x, out, scale, ratio=(3 / 4, 4 / 3)):
    """Per-sample RandomResizedCrop via roi_align (bilinear, area-averaged)."""
    B, _, H, W = x.shape
    dev = x.device
    area = _uniform(B, *scale, dev) * H * W
    logr = _uniform(B, math.log(ratio[0]), math.log(ratio[1]), dev)
    r = torch.exp(logr)
    w = torch.sqrt(area * r).clamp(max=W)
    h = torch.sqrt(area / r).clamp(max=H)
    x1 = torch.rand(B, device=dev) * (W - w)
    y1 = torch.rand(B, device=dev) * (H - h)
    boxes = torch.stack([torch.arange(B, device=dev, dtype=x.dtype), x1, y1, x1 + w, y1 + h], 1)
    return roi_align(x, boxes.to(x.dtype), output_size=(out, out), spatial_scale=1.0,
                     sampling_ratio=2, aligned=True)


def color_jitter(x, s, hue):
    B, dev = x.shape[0], x.device
    def f():
        return _uniform(B, 1 - s, 1 + s, dev).view(B, 1, 1, 1)
    x = (x * f()).clamp(0, 1)                                         # brightness
    m = _gray(x).mean((1, 2, 3), keepdim=True)
    x = ((x - m) * f() + m).clamp(0, 1)                               # contrast
    g = _gray(x)
    x = ((x - g) * f() + g).clamp(0, 1)                               # saturation
    if hue > 0:                                                       # hue: rotate in YIQ
        th = _uniform(B, -hue, hue, dev) * 2 * math.pi
        c, sn = torch.cos(th).view(B, 1, 1), torch.sin(th).view(B, 1, 1)
        T = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322],
                          [0.211, -0.523, 0.312]], device=dev, dtype=x.dtype)
        yiq = torch.einsum('ij,bjhw->bihw', T, x)
        i, q = yiq[:, 1], yiq[:, 2]
        yiq = torch.stack([yiq[:, 0], c * i - sn * q, sn * i + c * q], 1)
        x = torch.einsum('ij,bjhw->bihw', torch.linalg.inv(T), yiq).clamp(0, 1)
    return x


def augment(x_uint8, cfg_aug, out):
    """x_uint8: (B,3,S,S) uint8 on device -> normalised float (B,3,out,out)."""
    a = cfg_aug
    x = x_uint8.float() / 255.0
    B, dev = x.shape[0], x.device
    if a['scale_min'] < 1.0:
        x = random_resized_crop(x, out, (a['scale_min'], a['scale_max']))
    else:
        x = F.interpolate(x, size=(out, out), mode='bilinear', antialias=True, align_corners=False)
    if a['flip']:
        m = torch.rand(B, device=dev) < 0.5
        x = torch.where(m.view(B, 1, 1, 1), x.flip(3), x)
    if a['jitter_p'] > 0 and (a['jitter'] > 0 or a['hue'] > 0):
        m = torch.rand(B, device=dev) < a['jitter_p']
        x = torch.where(m.view(B, 1, 1, 1), color_jitter(x, a['jitter'], a['hue']), x)
    if a['gray_p'] > 0:
        m = torch.rand(B, device=dev) < a['gray_p']
        x = torch.where(m.view(B, 1, 1, 1), _gray(x).expand_as(x), x)
    return (x - MEAN) / STD


def eval_transform(x_uint8, out):
    return augment(x_uint8, PRESETS['none'], out)


def denormalize(x):
    return (x * STD + MEAN).clamp(0, 1)

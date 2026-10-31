"""Encoder (randomly initialised ResNet), SimCLR projection head, classifier."""
import torch
import torch.nn as nn
from torchvision import models as tvm

FEAT_DIMS = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048}


def make_encoder(arch='resnet18', small_stem=False):
    net = getattr(tvm, arch)(weights=None)
    if small_stem:
        net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net, FEAT_DIMS[arch]


class ProjectionHead(nn.Module):
    """SimCLR projection head g(h)."""
    def __init__(self, in_dim, out_dim=128, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                                 nn.ReLU(inplace=True), nn.Linear(hidden, out_dim))

    def forward(self, h):
        return self.net(h)


class Classifier(nn.Module):
    def __init__(self, arch, num_classes, small_stem=False, proj_dim=None, head_bn=False):
        super().__init__()
        self.encoder, d = make_encoder(arch, small_stem)
        self.head = nn.Sequential(nn.BatchNorm1d(d, affine=False), nn.Linear(d, num_classes)) if head_bn \
            else nn.Linear(d, num_classes)
        self.proj = ProjectionHead(d, proj_dim) if proj_dim else None   # joint training only

    def forward(self, x, return_feat=False):
        h = self.encoder(x)
        return (self.head(h), h) if return_feat else self.head(h)


def load_encoder_weights(model, path):
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('encoder', sd)
    model.encoder.load_state_dict(sd, strict=True)
    print(f'Loaded encoder weights from {path}')
    return model

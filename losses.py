import torch
import torch.nn.functional as F


def nt_xent(z1, z2, temperature=0.2):
    """SimCLR NT-Xent loss. For each of the 2B views the positive is the other view of
    the same image; the other 2B-2 views are negatives.
    """
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float('-inf'))
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)

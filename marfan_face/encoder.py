"""Encoder half of the conditional mesh VAE used for Marfan screening.

This module deliberately contains ONLY the encoder (mesh -> latent). The
decoder (latent -> mesh) is not part of this release: neither its weights nor
its up-sampling operators are shipped, so faces cannot be reconstructed from
the shared model. The encoder maps a registered facial mesh and its
(age, sex) conditioning to the deterministic posterior-mean latent code used
by the linear screener.

The forward path reproduces exactly the encoder used at training time
(`en_layers` of the original spiral-convolutional model), so the shared
weights load without modification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_add(src, index, dim, dim_size):
    """torch_scatter.scatter_add equivalent, implemented with index_add_ so the
    release has no torch_scatter dependency. Identical numerics."""
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.index_add_(dim, index, src)


def _pool(x, trans, dim=1):
    """Sparse mesh down-sampling (barycentric), matching the training model."""
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    return _scatter_add(out, row, dim, dim_size=trans.size(0))


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super().__init__()
        self.dim = dim
        self.indices = indices  # plain attribute (not a buffer) to match ckpt keys
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1)).view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1)).view(bs, n_nodes, -1)
        else:
            raise RuntimeError(f"x.dim() must be 2 or 3, got {x.dim()}")
        return self.layer(x)


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super().__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)

    def forward(self, x, down_transform):
        return _pool(F.elu(self.conv(x)), down_transform)


class SpiralEncoder(nn.Module):
    """Conditional spiral-convolutional encoder (mesh -> posterior mean).

    Structure mirrors `en_layers` of the training model: N spiral down-blocks
    followed by two linear heads (mu, logvar). Inference uses the mu head only.
    """

    def __init__(self, in_channels, out_channels, latent_size,
                 spiral_indices, down_transforms, cond_dim=0):
        super().__init__()
        self.out_channels = list(out_channels)
        self.cond_dim = cond_dim
        self.spiral_indices = spiral_indices          # list of LongTensor
        self.down_transforms = down_transforms        # list of sparse tensors
        self._flat = down_transforms[-1].size(0) * out_channels[-1]

        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            ic = in_channels if idx == 0 else out_channels[idx - 1]
            self.en_layers.append(
                SpiralEnblock(ic, out_channels[idx], spiral_indices[idx]))
        # two linear heads (mu at index -1, logvar at index -2), matching training
        self.en_layers.append(nn.Linear(self._flat + cond_dim, latent_size))
        self.en_layers.append(nn.Linear(self._flat + cond_dim, latent_size))

    @torch.no_grad()
    def encode(self, x, cond=None):
        """x: (B, V, 3) normalized vertices; cond: (B, cond_dim). Returns (B, d_z)."""
        for i in range(len(self.out_channels)):
            x = self.en_layers[i](x, self.down_transforms[i])
        x = x.reshape(-1, self._flat)
        if cond is not None:
            x = torch.cat([x, cond], dim=-1)
        return self.en_layers[-1](x)   # posterior mean (deterministic embedding)

    def to(self, device):
        super().to(device)
        self.spiral_indices = [s.to(device) for s in self.spiral_indices]
        self.down_transforms = [d.to(device) for d in self.down_transforms]
        for m in self.modules():
            if isinstance(m, SpiralConv):
                m.indices = m.indices.to(device)
        return self

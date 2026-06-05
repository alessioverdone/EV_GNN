"""
Mamba implementations for temporal forecasting.

Two versions are available in this file:
  - MambaModel  (active)  : uses mambapy — parallel scan, fast pure PyTorch.
  - MambaModelSequential  : commented out below — pure-PyTorch sequential scan
                            (reference/fallback, no extra deps but very slow).
"""

import torch
import torch.nn as nn
from mambapy.mamba import Mamba as _MambaBackbone, MambaConfig


# ---------------------------------------------------------------------------
# Active implementation — mambapy (parallel scan)
# ---------------------------------------------------------------------------

class MambaModel(nn.Module):
    """Mamba backbone (mambapy) with I/O projection head.

    Input:  (B*N, T, input_size)
    Output: (B*N, horizon * output_size)  — flat, ready to compare with y
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_layers: int,
        output_size: int,
        seq_len: int,
        horizon: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        cfg = MambaConfig(
            d_model=d_model,
            n_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
            pscan=True,       # vectorised parallel scan — fast
            use_cuda=False,   # no custom CUDA kernel needed
        )
        self.backbone     = _MambaBackbone(cfg)
        self.output_layer = nn.Linear(d_model * seq_len, output_size * horizon)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, T, input_size)
        x = self.input_proj(x)          # (B*N, T, d_model)
        x = self.backbone(x)            # (B*N, T, d_model)
        return self.output_layer(x.reshape(x.size(0), -1))


# ---------------------------------------------------------------------------
# Reference implementation — sequential scan, no extra deps (slow)
# ---------------------------------------------------------------------------

# import math
# import torch.nn.functional as F
# from einops import repeat
#
#
# class _MambaBlock(nn.Module):
#     def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
#         super().__init__()
#         self.d_model  = d_model
#         self.d_state  = d_state
#         self.d_inner  = d_model * expand
#         self.dt_rank  = math.ceil(d_model / 16)
#         self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
#         self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner,
#                                   kernel_size=d_conv, groups=self.d_inner,
#                                   padding=d_conv - 1, bias=True)
#         self.x_proj   = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
#         self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)
#         A = repeat(torch.arange(1, d_state + 1, dtype=torch.float),
#                    'n -> d n', d=self.d_inner)
#         self.A_log = nn.Parameter(torch.log(A))
#         self.D     = nn.Parameter(torch.ones(self.d_inner))
#         self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
#
#     def _ssm(self, u):
#         B_size, L, _ = u.shape
#         A = -torch.exp(self.A_log)
#         D = self.D
#         x_dbl = self.x_proj(u)
#         delta, B_mat, C_mat = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
#         delta = F.softplus(self.dt_proj(delta))
#         dA  = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
#         dBu = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * u.unsqueeze(-1)
#         h = torch.zeros(B_size, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
#         ys = []
#         for i in range(L):
#             h = dA[:, i] * h + dBu[:, i]
#             ys.append((h * C_mat[:, i].unsqueeze(1)).sum(-1))
#         return torch.stack(ys, dim=1) + u * D
#
#     def forward(self, x):
#         B, L, _ = x.shape
#         x_part, z = self.in_proj(x).chunk(2, dim=-1)
#         x_part = self.conv1d(x_part.transpose(1, 2))[:, :, :L].transpose(1, 2)
#         x_part = F.silu(x_part)
#         return self.out_proj(self._ssm(x_part) * F.silu(z))
#
#
# class _MambaResidualBlock(nn.Module):
#     def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
#         super().__init__()
#         self.mamba = _MambaBlock(d_model, d_state, d_conv, expand)
#         self.norm  = nn.RMSNorm(d_model)
#
#     def forward(self, x):
#         return self.mamba(self.norm(x)) + x
#
#
# class MambaModelSequential(nn.Module):
#     """Fallback: same interface as MambaModel but uses a sequential scan loop.
#     Correct but slow — O(L) Python overhead per batch.
#     """
#     def __init__(self, input_size, d_model, num_layers, output_size,
#                  seq_len, horizon, d_state=16, d_conv=4, expand=2):
#         super().__init__()
#         self.input_proj  = nn.Linear(input_size, d_model)
#         self.layers      = nn.ModuleList([
#             _MambaResidualBlock(d_model, d_state, d_conv, expand)
#             for _ in range(num_layers)
#         ])
#         self.norm         = nn.RMSNorm(d_model)
#         self.output_layer = nn.Linear(d_model * seq_len, output_size * horizon)
#         nn.init.xavier_uniform_(self.output_layer.weight)
#
#     def forward(self, x):
#         x = self.input_proj(x)
#         for layer in self.layers:
#             x = layer(x)
#         x = self.norm(x)
#         return self.output_layer(x.reshape(x.size(0), -1))

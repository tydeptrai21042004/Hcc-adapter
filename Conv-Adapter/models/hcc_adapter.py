import torch
import torch.nn as nn

class HartleyCosineConv2d(nn.Module):
    """
    Even-symmetric shift aggregation along H or W with circular/zero padding.
    Returns a *residual* tensor; caller does: out + scale * residual(out).
    """
    def __init__(self, h=1, num_shifts=1, axis='h', padding='circular',
                 per_channel=True, use_center=True):
        super().__init__()
        assert axis in ('h', 'w')
        assert padding in ('circular', 'zero')
        self.h = int(h)
        self.M = int(num_shifts)
        self.axis = axis
        self.padding = padding
        self.per_channel = per_channel
        self.use_center = use_center
        self.weight = None  # lazily built as (C,K) or (K,)

    def _ensure_weight(self, C, device):
        # IMPORTANT: keep parameter in FP32 so grads are FP32 (needed by GradScaler)
        K = (1 + 2 * self.M) if self.use_center else (2 * self.M)
        shape = (C, K) if self.per_channel else (K,)
        if (self.weight is None) or (tuple(self.weight.shape) != shape):
            w = torch.empty(shape, device=device, dtype=torch.float32)
            nn.init.xavier_uniform_(w)
            self.weight = nn.Parameter(w)
        else:
            if self.weight.device != device:
                self.weight.data = self.weight.data.to(device=device)

    def _shift_zero(self, x, s, dim):
        if s == 0:
            return x
        if dim == 2:  # H
            if s > 0:
                pad = torch.zeros_like(x)[:, :, :s, :]
                core = x[:, :, :-s, :]
                return torch.cat([pad, core], dim=2)
            else:
                s = -s
                pad = torch.zeros_like(x)[:, :, -s:, :]
                core = x[:, :, s:, :]
                return torch.cat([core, pad], dim=2)
        else:         # W
            if s > 0:
                pad = torch.zeros_like(x)[:, :, :, :s]
                core = x[:, :, :, :-s]
                return torch.cat([pad, core], dim=3)
            else:
                s = -s
                pad = torch.zeros_like(x)[:, :, :, -s:]
                core = x[:, :, :, s:]
                return torch.cat([core, pad], dim=3)

    def forward(self, x):
        """
        x: (N,C,H,W)  ->  residual: (N,C,H,W)
        """
        C = x.size(1)
        self._ensure_weight(C, device=x.device)

        # Build even-symmetric shifts: [0, +h, -h, +2h, -2h, ...]
        shifts = [0] if self.use_center else []
        for m in range(1, self.M + 1):
            s = m * self.h
            shifts.extend([+s, -s])

        dim = 2 if self.axis == 'h' else 3
        bank = []
        for s in shifts:
            if self.padding == 'circular':
                bank.append(torch.roll(x, shifts=s, dims=dim))
            else:
                bank.append(self._shift_zero(x, s, dim))
        X = torch.stack(bank, dim=-1)  # (N,C,H,W,K)

        # Compute with a casted view (keeps leaf param in FP32 for FP32 grads)
        if self.per_channel:
            Wc = self.weight.view(1, C, 1, 1, -1)
        else:
            Wc = self.weight.view(1, 1, 1, 1, -1)
        if Wc.dtype != X.dtype:
            Wc = Wc.to(X.dtype)

        residual = (X * Wc).sum(dim=-1)  # (N,C,H,W)
        return residual

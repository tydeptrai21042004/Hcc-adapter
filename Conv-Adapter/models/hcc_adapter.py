# file: models/hcc_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HCCAdapter(nn.Module):
    """
    Hartley–Cosine (even) shift aggregation via depthwise dilated conv.
    - Axis: 'h', 'w', or 'hw' (sum).
    - Tie ±m weights; learn per-channel alphas (center + M side taps).
    - Optional pointwise bottleneck mixing + BN.
    - Reflect padding by default to avoid wrap artifacts.
    """
    def __init__(
        self, C, M=1, h=1, axis='hw',
        per_channel=True, tie_sym=True,
        use_pw=True, pw_ratio=8, use_bn=True,
        residual_scale=1.0, gate_init=0.1,
        padding_mode='reflect'
    ):
        super().__init__()
        assert axis in ('h','w','hw')
        self.C = C
        self.M = int(M)
        self.h = int(h)
        self.axis = axis
        self.tie_sym = tie_sym
        self.per_channel = per_channel
        self.padding_mode = padding_mode
        self.residual_scale = residual_scale

        K = 2*M + 1  # kernel length along axis

        # Learn alpha coefficients: center + M side (shared or per-channel)
        ncoef = M + 1
        if per_channel:
            self.alpha = nn.Parameter(torch.zeros(C, ncoef))
        else:
            self.alpha = nn.Parameter(torch.zeros(ncoef))

        # Identity-safe init: center ≈ 1, sides ≈ 0
        with torch.no_grad():
            if per_channel:
                self.alpha[:, 0].fill_(1.0)
            else:
                self.alpha[0] = 1.0

        # Optional channel mixing (DW -> PW bottleneck -> PW expand)
        self.use_pw = use_pw
        if use_pw:
            hid = max(1, C // pw_ratio)
            self.pw = nn.Sequential(
                nn.Conv2d(C, hid, 1, bias=False),
                nn.BatchNorm2d(hid) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid, C, 1, bias=False),
                nn.BatchNorm2d(C) if use_bn else nn.Identity(),
            )
        else:
            self.pw = nn.Identity()

        # Learnable global gate (can switch to per-channel gate if desired)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def _build_even_kernel_1d(self, device, dtype):
        """
        Build symmetric 1D kernel of length K = 2M+1 from alpha (center + M sides).
        If per_channel=True, returns weight shape for depthwise conv (C,1,K).
        Else returns (1,1,K) and we expand to groups=C.
        """
        M, C = self.M, self.C
        K = 2*M + 1

        # Compose full even kernel from alpha [alpha0, alpha1..alphaM]
        # w[k] = alpha0 at center; alpha_m at ±m positions
        if self.per_channel:
            # (C, K)
            w = torch.zeros(C, K, device=device, dtype=dtype)
            center = M
            w[:, center] = self.alpha[:, 0]
            for m in range(1, M+1):
                if self.tie_sym:
                    w[:, center - m] = self.alpha[:, m]
                    w[:, center + m] = self.alpha[:, m]
                else:
                    # If you later store separate +/−, split here
                    w[:, center - m] = self.alpha[:, m]
                    w[:, center + m] = self.alpha[:, m]
            # normalize (optional but recommended)
            s = w.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
            w = w / s
            # depthwise weight (C,1,K)
            return w.unsqueeze(1)
        else:
            # (K,)
            w = torch.zeros(K, device=device, dtype=dtype)
            center = M
            w[center] = self.alpha[0]
            for m in range(1, M+1):
                val = self.alpha[m]
                if self.tie_sym:
                    w[center-m] = val
                    w[center+m] = val
                else:
                    w[center-m] = val
                    w[center+m] = val
            s = w.abs().sum().clamp_min(1e-6)
            w = w / s
            # expand to depthwise (C,1,K)
            return w.view(1,1,K).repeat(self.C, 1, 1)

    def _pad(self, x, pad_h, pad_w):
        if self.padding_mode == 'reflect':
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        elif self.padding_mode == 'replicate':
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        else:
            # 'zeros'
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        w1d = self._build_even_kernel_1d(x.device, x.dtype)  # (C,1,K)

        y = 0
        K = 2*self.M + 1
        if 'h' in self.axis:
            # depthwise conv along height -> kernel size (K,1), dilation (h,1)
            wh = w1d.view(self.C, 1, K, 1)
            xh = self._pad(x, pad_h=self.M*self.h, pad_w=0)
            yh = F.conv2d(xh, wh, bias=None, stride=1, padding=0,
                          dilation=(self.h, 1), groups=self.C)
            y = y + yh

        if 'w' in self.axis:
            # depthwise conv along width -> kernel size (1,K), dilation (1,h)
            ww = w1d.view(self.C, 1, 1, K)
            xw = self._pad(x, pad_h=0, pad_w=self.M*self.h)
            yw = F.conv2d(xw, ww, bias=None, stride=1, padding=0,
                          dilation=(1, self.h), groups=self.C)
            y = y + yw

        # Residual + optional PW mixing + gate
        y = self.pw(y)
        return x + self.residual_scale * self.gate * y

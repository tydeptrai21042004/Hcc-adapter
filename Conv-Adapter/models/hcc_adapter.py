import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- helpers ----------

def _pad_shift(x, s: int, axis: str, mode: str):
    """
    Shift by s pixels along axis ('h' or 'w') using padding mode.
    mode: 'circular', 'reflect', 'replicate', 'zeros'
    x: (N, C, H, W)
    """
    if s == 0:
        return x

    if mode == 'circular':
        dim = 2 if axis == 'h' else 3
        return torch.roll(x, shifts=s, dims=dim)

    # F.pad uses (left, right, top, bottom); last two dims are H, W
    if axis == 'h':
        if s > 0:
            # shift down by s => pad top by s, crop bottom
            pad = (0, 0, s, 0)
            if mode == 'zeros':
                xpad = F.pad(x, pad, mode='constant', value=0)
            else:
                xpad = F.pad(x, pad, mode=mode)
            return xpad[:, :, :x.size(2), :]
        else:
            s = -s  # shift up
            pad = (0, 0, 0, s)
            if mode == 'zeros':
                xpad = F.pad(x, pad, mode='constant', value=0)
            else:
                xpad = F.pad(x, pad, mode=mode)
            return xpad[:, :, s:, :]
    else:  # axis == 'w'
        if s > 0:
            # shift right by s => pad left by s, crop right
            pad = (s, 0, 0, 0)
            if mode == 'zeros':
                xpad = F.pad(x, pad, mode='constant', value=0)
            else:
                xpad = F.pad(x, pad, mode=mode)
            return xpad[:, :, :, :x.size(3)]
        else:
            s = -s  # shift left
            pad = (0, s, 0, 0)
            if mode == 'zeros':
                xpad = F.pad(x, pad, mode='constant', value=0)
            else:
                xpad = F.pad(x, pad, mode=mode)
            return xpad[:, :, :, s:]


def _build_shifts(M: int, h: int, use_center: bool):
    """
    Returns ordered list: [0, +h, -h, +2h, -2h, ...] if use_center else [+h,-h,...]
    """
    shifts = []
    if use_center:
        shifts.append(0)
    for m in range(1, M + 1):
        s = m * h
        shifts.extend([+s, -s])
    return shifts


# ---------- core op ----------

class HartleyCosineConv2d(nn.Module):
    """
    Even-shift aggregation along one axis.

    Performs: sum_k w[..., k] * shift(x, s_k, axis)
    - per_channel: weights have shape (C, K), else (K,)
    - tie_sym: if True, +mh and -mh share the same coefficient
               (learn only center + M positives; expand internally)
    """
    def __init__(
        self,
        C: int,
        M: int = 1,
        h: int = 1,
        axis: str = 'h',                 # 'h' or 'w'
        padding_mode: str = 'reflect',   # 'reflect' | 'replicate' | 'zeros' | 'circular'
        per_channel: bool = True,
        use_center: bool = True,
        tie_sym: bool = True,
        init_scale_noncenter: float = 0.0,   # init small
        init_center: float = 0.0,            # residual path has its own gate; keep center small by default
    ):
        super().__init__()
        assert axis in ('h', 'w')
        assert padding_mode in ('reflect', 'replicate', 'zeros', 'circular')

        self.C = int(C)
        self.M = int(M)
        self.h = int(h)
        self.axis = axis
        self.padding_mode = padding_mode
        self.per_channel = per_channel
        self.use_center = use_center
        self.tie_sym = tie_sym

        # logical sizes
        self.K = (1 + 2 * M) if use_center else (2 * M)
        self.K_tied = (1 + M) if use_center else M  # center + M positives

        if self.tie_sym:
            # learn logits for center & +m only (broadcast to +/-)
            if per_channel:
                self.alpha = nn.Parameter(torch.zeros(self.C, self.K_tied))
            else:
                self.alpha = nn.Parameter(torch.zeros(self.K_tied))
            # init
            if use_center:
                if per_channel:
                    self.alpha.data[:, 0] = init_center
                    if self.K_tied > 1:
                        self.alpha.data[:, 1:] = init_scale_noncenter
                else:
                    self.alpha.data[0] = init_center
                    if self.K_tied > 1:
                        self.alpha.data[1:] = init_scale_noncenter
            else:
                if per_channel:
                    self.alpha.data[:, :] = init_scale_noncenter
                else:
                    self.alpha.data[:] = init_scale_noncenter
        else:
            # learn independent weights for 0 / +m / -m
            if per_channel:
                self.alpha = nn.Parameter(torch.zeros(self.C, self.K))
                self.alpha.data[:] = init_scale_noncenter
                if use_center:
                    self.alpha.data[:, 0] = init_center
            else:
                self.alpha = nn.Parameter(torch.zeros(self.K))
                self.alpha.data[:] = init_scale_noncenter
                if use_center:
                    self.alpha.data[0] = init_center

        self.register_buffer('shifts', torch.tensor(_build_shifts(M, h, use_center), dtype=torch.int64))

    def _expand_tied(self):
        """
        From (C, 1+M) -> (C, K) or (1+M) -> (K) by mirroring +m to +/-m.
        Order: [0, +h, -h, +2h, -2h, ...]
        """
        if not self.tie_sym:
            return self.alpha

        if self.per_channel:
            C = self.alpha.shape[0]
            if self.use_center:
                center = self.alpha[:, 0:1]  # (C,1)
                pos = self.alpha[:, 1:]      # (C,M)
            else:
                center = None
                pos = self.alpha[:, :]       # (C,M)

            pieces = []
            if center is not None:
                pieces.append(center)
            for m in range(pos.shape[1]):
                a = pos[:, m:m+1]  # (C,1)
                pieces.append(a)   # +mh
                pieces.append(a)   # -mh (tied)
            return torch.cat(pieces, dim=1)  # (C, K)
        else:
            if self.use_center:
                center = self.alpha[0:1]
                pos = self.alpha[1:]
            else:
                center = None
                pos = self.alpha
            pieces = []
            if center is not None:
                pieces.append(center)
            for m in range(pos.shape[0]):
                a = pos[m:m+1]
                pieces.append(a)
                pieces.append(a)
            return torch.cat(pieces, dim=0)  # (K,)

    def forward(self, x):
        """
        x: (N, C, H, W)  ->  y: (N, C, H, W)
        """
        N, C, H, W = x.shape
        assert C == self.C, f'Channel mismatch: got {C}, expected {self.C}'

        # get full alpha (expand if tied)
        alpha_full = self._expand_tied()  # (C,K) or (K,)
        # broadcast to (1,C,1,1,K) or (1,1,1,1,K)
        if self.per_channel:
            Wcoef = alpha_full.view(1, C, 1, 1, self.K)
        else:
            Wcoef = alpha_full.view(1, 1, 1, 1, self.K)

        # build bank along axis
        bank = []
        for idx, s in enumerate(self.shifts.tolist()):
            bank.append(_pad_shift(x, s, self.axis, self.padding_mode))
        Xbank = torch.stack(bank, dim=-1)  # (N, C, H, W, K)

        # weighted sum across K
        y = (Xbank * Wcoef).sum(dim=-1)    # (N, C, H, W)
        return y


# ---------- full adapter (with bottleneck, norm, gate, axis='hw' support) ----------

class HCCAdapter(nn.Module):
    """
    Full residual adapter:
      y = x + sigma(gate) * PW2( Norm( HCC( Norm( PW1(x) ) ) ) )

    Options:
      - axis: 'h' | 'w' | 'hw' (sum of two HCC ops)
      - per_channel/tie_sym/use_center per HartleyCosineConv2d
      - padding_mode: reflect (default), replicate, zeros, circular
      - use_pw: enables 1x1 bottleneck around the HCC op
      - pw_ratio: C -> C//r -> C
      - residual_scale: multiply residual before adding to x (global scalar)
      - gate_init: initialize gate logits so sigma(gate) starts small (e.g., 0.1)
    """
    def __init__(
        self,
        C: int,
        M: int = 1,
        h: int = 1,
        axis: str = 'hw',
        per_channel: bool = True,
        tie_sym: bool = True,
        use_center: bool = True,
        padding_mode: str = 'reflect',
        use_pw: bool = True,
        pw_ratio: int = 8,
        norm_eps: float = 1e-5,
        residual_scale: float = 1.0,
        gate_init: float = 0.1,
    ):
        super().__init__()
        self.C = int(C)
        self.axis = axis
        self.use_pw = use_pw
        self.residual_scale = residual_scale

        # pre norm
        self.bn1 = nn.BatchNorm2d(C, eps=norm_eps)

        # pointwise bottleneck (optional)
        if use_pw:
            Cmid = max(1, C // pw_ratio)
            self.pw1 = nn.Conv2d(C, Cmid, kernel_size=1, bias=False)
            self.act = nn.GELU()
            self.bn_mid = nn.BatchNorm2d(Cmid, eps=norm_eps)
            # HCC works on Cmid channels
            self.hcc_h = HartleyCosineConv2d(
                C=Cmid, M=M, h=h,
                axis='h' if axis in ('h', 'hw') else 'w',
                padding_mode=padding_mode,
                per_channel=per_channel,
                use_center=use_center,
                tie_sym=tie_sym,
                init_center=0.0,
                init_scale_noncenter=0.0,
            )
            if axis == 'hw':
                self.hcc_w = HartleyCosineConv2d(
                    C=Cmid, M=M, h=h,
                    axis='w', padding_mode=padding_mode,
                    per_channel=per_channel,
                    use_center=use_center,
                    tie_sym=tie_sym,
                    init_center=0.0,
                    init_scale_noncenter=0.0,
                )
            else:
                self.hcc_w = None
            self.bn2 = nn.BatchNorm2d(Cmid, eps=norm_eps)
            self.pw2 = nn.Conv2d(Cmid, C, kernel_size=1, bias=False)
        else:
            # operate at full C
            self.hcc_h = HartleyCosineConv2d(
                C=C, M=M, h=h,
                axis='h' if axis in ('h', 'hw') else 'w',
                padding_mode=padding_mode,
                per_channel=per_channel,
                use_center=use_center,
                tie_sym=tie_sym,
                init_center=0.0,
                init_scale_noncenter=0.0,
            )
            if axis == 'hw':
                self.hcc_w = HartleyCosineConv2d(
                    C=C, M=M, h=h,
                    axis='w', padding_mode=padding_mode,
                    per_channel=per_channel,
                    use_center=use_center,
                    tie_sym=tie_sym,
                    init_center=0.0,
                    init_scale_noncenter=0.0,
                )
            else:
                self.hcc_w = None
            self.bn2 = nn.BatchNorm2d(C, eps=norm_eps)
            self.pw1 = None
            self.pw2 = None
            self.act = nn.GELU()

        # post norm
        self.bn_out = nn.BatchNorm2d(C, eps=norm_eps)

        # residual gate (logits -> sigmoid)
        # gate_init ~ 0.1 => sigmoid^-1(0.1) ~ -2.197
        g0 = torch.tensor([torch.logit(torch.tensor(gate_init))], dtype=torch.float32)
        self.gate_logit = nn.Parameter(g0.clone())

        # mark so main.py hook knows to return full output (x + residual) already combined
        self.is_hcc_adapter = True

    def forward(self, x):
        identity = x
        out = self.bn1(x)

        if self.use_pw:
            out = self.pw1(out)
            out = self.bn_mid(out)
            out = self.act(out)

            y = self.hcc_h(out)
            if self.hcc_w is not None:
                y = y + self.hcc_w(out)

            y = self.bn2(y)
            y = self.pw2(y)
        else:
            y = self.hcc_h(out)
            if self.hcc_w is not None:
                y = y + self.hcc_w(out)
            y = self.bn2(y)

        y = self.bn_out(self.act(y))
        gate = torch.sigmoid(self.gate_logit)
        return identity + self.residual_scale * gate * y

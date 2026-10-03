import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEState:
    """Shared mutable context for every MoE-LoRA layer (gate, pad mask, aux + stats collection)."""
    def __init__(self, gate):
        self.gate = gate
        self.token_mask = None        # [B,T] bool (True = real token); set by caller each forward
        self.collect_aux = False
        self.collect_stats = False
        self.aux_terms, self.stats = [], []
    def reset(self):
        self.aux_terms, self.stats = [], []
    def aux_loss(self):
        return sum(self.aux_terms) / len(self.aux_terms) if self.aux_terms else 0.0

class FixedTopKGate:
    """Eq. 1: fixed top-k by router prob, renormalized. Returns (weights [...,N], k [...])."""
    name = "fixed_topk"
    def __init__(self, k): self.k = k
    def __call__(self, probs, layer, z):
        vals, idx = probs.topk(self.k, dim=-1)
        vals = vals / vals.sum(-1, keepdim=True)
        w = torch.zeros_like(probs).scatter(-1, idx, vals)
        k = torch.full(probs.shape[:-1], self.k, dtype=torch.long, device=probs.device)
        return w, k

class MoELoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, state: MoEState, N, r, alpha, dropout, name="", router_std=0.01):
        super().__init__()
        self.base, self.state, self.N, self.r, self.name = base, state, N, r, name
        self.scale = alpha / r
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.empty(N, r, d_in))
        self.B = nn.Parameter(torch.zeros(N, d_out, r))       # zero-init: starts identical to base
        self.router = nn.Parameter(torch.randn(N, d_in) * router_std)
        self.drop = nn.Dropout(dropout)                       # inert in eval()
        for i in range(N):
            nn.init.kaiming_uniform_(self.A.data[i], a=math.sqrt(5))

    def _mask(self, probs):
        m = self.state.token_mask
        return torch.ones(probs.shape[:-1], dtype=torch.bool, device=probs.device) if m is None else m

    def forward(self, x):
        y = self.base(x)
        st, dt = self.state, x.dtype
        xd = self.drop(x)
        probs = F.softmax(F.linear(x, self.router.to(dt)).float(), dim=-1)      # [...,N]
        z = torch.einsum("...d,nrd->...nr", xd, self.A.to(dt))                  # [...,N,r]
        w, k = st.gate(probs, self, z)                                          # w sums to 1 over admitted set
        zw = (z * w.unsqueeze(-1).to(dt)).flatten(-2)                           # [..., N*r]
        Bcat = self.B.to(dt).permute(1, 0, 2).reshape(self.base.out_features, -1)
        delta = F.linear(zw, Bcat) * self.scale

        if st.collect_aux and self.training:
            m = self._mask(probs)
            P = probs[m].mean(0)
            assigned = (w[m] > 0).float()
            f = (assigned.sum(0) / assigned.sum()).detach()
            st.aux_terms.append(self.N * (f * P).sum())                         # N * sum_i f_i P_i
        if st.collect_stats:
            with torch.no_grad():
                m = self._mask(probs)
                st.stats.append((self.name, torch.bincount(k[m].long(), minlength=self.N + 1).cpu()))
        return y + delta

def inject_moe_lora(model, state, N=16, r=8, alpha=16, dropout=0.05, router_std=0.01,
                    targets=("gate_proj", "up_proj", "down_proj")):
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for li, layer in enumerate(model.model.layers):
        for t in targets:
            base = getattr(layer.mlp, t)
            new = MoELoRALinear(base, state, N, r, alpha, dropout, name=f"L{li}.{t}", router_std=router_std)
            setattr(layer.mlp, t, new.to(base.weight.device))   # new params stay fp32, trainable
            n += 1
    return n

_TRAINABLE_SUFFIX = (".A", ".B", ".router")

def save_moe_checkpoint(model, path, extra=None):
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    assert sd and all(n.endswith(_TRAINABLE_SUFFIX) for n in sd), "unexpected trainable params"
    torch.save({"state": sd, "extra": extra or {}}, path)

def load_moe_checkpoint(model, path):
    """strict=False, then LOUD assertions (context doc section 8)."""
    ck = torch.load(path, map_location="cpu")
    res = model.load_state_dict(ck["state"], strict=False)
    assert not res.unexpected_keys, f"UNEXPECTED KEYS: {res.unexpected_keys[:5]}"
    bad = [k for k in res.missing_keys if k.endswith(_TRAINABLE_SUFFIX)]
    assert not bad, f"MISSING ROUTER/EXPERT KEYS: {bad[:5]}"
    print(f"[ckpt] loaded {len(ck['state'])} tensors | unexpected=0 | missing (frozen base only)={len(res.missing_keys)}")
    return ck["extra"]
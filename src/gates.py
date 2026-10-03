import torch
from common import PAPER
from moe_lora import FixedTopKGate

class CAREGate:
    """CARE expert admission (Eqs. 2-4, Alg. 1), test-time drop-in for MoELoRALinear.
    Disagreement D (Eq. 2) is computed EXACTLY in output space via the Gram matrices G_nm = B_n^T B_m (r x r):
      ||e_i||^2 = z_i^T G_ii z_i ,  ||ebar||^2 = sum_ij pt_i pt_j z_i^T G_ij z_j
      D = (sum pt_i||e_i||^2 - ||ebar||^2) / (sum pt_i||e_i||^2 + eps*d_out)
    == mean_c(sum_i pt_i (e_ic - ebar_c)^2) / ( sum_i pt_i mean_c(e_ic^2) + eps )   [normalizer = mean of INDIVIDUAL squared
    magnitudes, NOT the magnitude of the combined output; context doc Sec. 5]. The LoRA scale alpha/r cancels in the ratio."""
    name = "care"

    def __init__(self, tau, k_min=1, k_max=8, gamma=2, delta=0.55, eps=1e-8, record=False):
        assert 0.0 < tau <= 1.0 and 1 <= k_min <= k_max and 0.0 <= delta < 1.0
        self.tau, self.k_min, self.k_max, self.gamma, self.delta, self.eps = tau, k_min, k_max, gamma, delta, eps
        self.record, self.trace = record, []

    @staticmethod
    def gram(layer):
        c = getattr(layer, "_gram_cache", None)
        ver = layer.B._version
        if c is None or c[0] != ver or layer.training:
            B = layer.B.detach().float()                       # [N, d_out, r]
            c = (ver, torch.einsum("ndr,mds->nmrs", B, B))     # [N, N, r, r]
            layer._gram_cache = c
        return c[1]

    def disagreement(self, layer, z, wt):
        """z: [..., N, r] (A_i x), wt: [..., N] weights over the admitted set (sum to 1, zero elsewhere)."""
        G = self.gram(layer)
        z = z.float()
        Gz = torch.einsum("nmrs,...ms->...nmr", G, z)          # G_nm z_m
        M = torch.einsum("...nr,...nmr->...nm", z, Gz)         # z_n^T G_nm z_m  (= e_n . e_m)
        num_mag = (wt * torch.diagonal(M, dim1=-2, dim2=-1)).sum(-1)
        mean_sq = torch.einsum("...n,...nm,...m->...", wt, M, wt)
        D = (num_mag - mean_sq) / (num_mag + self.eps * layer.base.out_features)
        return D.clamp(0.0, 1.0)

    def __call__(self, probs, layer, z):
        N = probs.shape[-1]
        sp, idx = probs.sort(dim=-1, descending=True)
        cum = sp.cumsum(-1)
        k_nu = ((cum < self.tau).sum(-1) + 1).clamp(max=N)                     # Eq. 3
        rank = torch.arange(N, device=probs.device)
        adm = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, idx, rank < k_nu.unsqueeze(-1))
        wt = probs * adm
        wt = wt / wt.sum(-1, keepdim=True)
        D = self.disagreement(layer, z, wt)                                    # Eq. 2 over the nucleus set S
        rho = ((D - self.delta) / (1.0 - self.delta)).clamp(min=0.0)
        k = (k_nu + torch.ceil(self.gamma * rho).long()).clamp(self.k_min, self.k_max)   # Eq. 4, strict clip
        keep = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, idx, rank < k.unsqueeze(-1))
        w = probs * keep
        w = w / w.sum(-1, keepdim=True)                                        # Eq. 1 renormalization
        if self.record:
            m = layer.state.token_mask
            m = torch.ones(k.shape, dtype=torch.bool, device=k.device) if m is None else m
            self.trace.append(dict(name=layer.name, n=int(m.sum()),
                                   k_nu=torch.bincount(k_nu[m], minlength=N + 1).cpu(),
                                   k=torch.bincount(k[m], minlength=N + 1).cpu(),
                                   D_hist=torch.histc(D[m].float(), bins=20, min=0, max=1).cpu()))
        return w, k

def make_gate(name, **kw):
    if name == "fixed":
        return FixedTopKGate(kw.get("k", PAPER["train_k"]))
    if name == "care":
        return CAREGate(tau=kw["tau"], k_min=kw.get("k_min", PAPER["k_min"]), k_max=kw.get("k_max", PAPER["k_max"]),
                        gamma=kw.get("gamma", PAPER["gamma"]), delta=kw.get("delta", PAPER["delta"]),
                        record=kw.get("record", False))
    raise ValueError(name)
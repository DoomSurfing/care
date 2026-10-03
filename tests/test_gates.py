import torch, torch.nn as nn, torch.nn.functional as F
from moe_lora import MoEState, MoELoRALinear, FixedTopKGate
from gates import CAREGate, make_gate
torch.manual_seed(0)

def make_layer(N=8, r=4, d_in=32, d_out=64):
    st = MoEState(FixedTopKGate(2))
    L = MoELoRALinear(nn.Linear(d_in, d_out, bias=False), st, N, r, alpha=2 * r, dropout=0.0, name="t")
    with torch.no_grad(): L.B.normal_(0, 0.3)
    L.eval()
    return st, L

def zof(L, x): return torch.einsum("...d,nrd->...nr", x, L.A)
P8 = torch.tensor([[[0.7, 0.2, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002]]])
st, L = make_layer(); x = torch.randn(1, 1, 32); z = zof(L, x)

# T1: nucleus size k_nu (gamma=0 -> no extension)
for tau, want in [(0.65, 1), (0.85, 2), (0.94, 3), (0.999, 8)]:
    _, k = CAREGate(tau, k_max=8, gamma=0)(P8, L, z)
    assert k.item() == want, (tau, k.item(), want)
print("T1 nucleus k_nu: ok")

# T2: strict k_min / k_max
assert CAREGate(0.999, k_max=6, gamma=0)(P8, L, z)[1].item() == 6
assert CAREGate(0.01, k_min=2, gamma=0)(P8, L, z)[1].item() == 2
print("T2 k_min/k_max clip: ok")

# T3: k_bar non-decreasing in tau (gamma=0), on random routers
probs = F.softmax(torch.randn(2, 200, 8) * 2, -1); zz = zof(L, torch.randn(2, 200, 32))
prev = 0.0
for tau in [0.2, 0.4, 0.6, 0.8, 0.95, 1.0]:
    kb = CAREGate(tau, gamma=0)(probs, L, zz)[1].float().mean().item()
    assert kb >= prev - 1e-9; prev = kb
print("T3 k_bar monotone in tau: ok")

# T4: Gram-based D == naive per-coordinate Eq. 2 with explicit expert outputs
x = torch.randn(3, 7, 32); z = zof(L, x); p = F.softmax(F.linear(x, L.router), -1)
top = p.topk(3, -1).indices; wt = torch.zeros_like(p).scatter(-1, top, p.gather(-1, top)); wt = wt / wt.sum(-1, keepdim=True)
g = CAREGate(0.9); D = g.disagreement(L, z, wt)
e = torch.einsum("...nr,ndr->...nd", z, L.B)                         # [3,7,N,d_out] explicit outputs
ebar = (wt[..., None] * e).sum(-2)
numer = (wt[..., None] * (e - ebar[..., None, :]) ** 2).sum(-2).mean(-1)
norm = (wt * (e ** 2).mean(-1)).sum(-1)
Dn = numer / (norm + g.eps)
assert torch.allclose(D, Dn, atol=1e-4), (D - Dn).abs().max()
print("T4 Gram D matches naive Eq.2: ok  (D range %.3f..%.3f)" % (D.min(), D.max()))

# T5/T6: cancellation-safe normalizer. Opposite experts -> D=1 ; identical experts -> D=0
st, L2 = make_layer()
with torch.no_grad(): L2.A[1] = L2.A[0]; L2.B[1] = -L2.B[0]
x = torch.randn(1, 1, 32); z = zof(L2, x)
w2 = torch.zeros(1, 1, 8); w2[..., 0] = w2[..., 1] = 0.5
assert abs(CAREGate(0.9).disagreement(L2, z, w2).item() - 1.0) < 1e-4
with torch.no_grad(): L2.B[1] = L2.B[0]
assert CAREGate(0.9).disagreement(L2, z, w2).item() < 1e-4
print("T5/T6 opposite -> D=1, identical -> D=0: ok")

# T7: extension fires: k_nu=2, D=1 -> rho=1 -> +ceil(2*1)=2 -> k=4 ; T8: k_nu=1 => D=0 => never extends
with torch.no_grad(): L2.B[1] = -L2.B[0]
p7 = torch.tensor([[[0.45, 0.45, 0.04, 0.03, 0.01, 0.01, 0.005, 0.005]]])
w7, k7 = CAREGate(0.85, gamma=2, delta=0.55)(p7, L2, z)
assert k7.item() == 4 and (w7 > 0).sum().item() == 4
p8 = torch.tensor([[[0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002]]])
assert CAREGate(0.8, gamma=2, delta=0.55)(p8, L2, z)[1].item() == 1
print("T7 extension fires (k=4); T8 k_nu=1 never extends (D==0 by construction): ok")

# T9: output weights: sum to 1, exactly k nonzero, chosen set == top-k by router prob
probs = F.softmax(torch.randn(4, 50, 8) * 1.5, -1); zz = zof(L, torch.randn(4, 50, 32))
w, k = CAREGate(0.7)(probs, L, zz)
assert torch.allclose(w.sum(-1), torch.ones(4, 50), atol=1e-5) and ((w > 0).sum(-1) == k).all()
kth = probs.sort(-1, descending=True).values.gather(-1, (k - 1).unsqueeze(-1))
assert ((probs >= kth) == (w > 0)).all()
print("T9 weights sum to 1, |S|==k, S==top-k: ok")

# T10: flat router with N=16 respects k_max=8; and k VARIES across tokens on realistic routers
st16, L16 = make_layer(N=16)
flat = torch.full((1, 1, 16), 1 / 16); z16 = zof(L16, torch.randn(1, 1, 32))
w, k = CAREGate(0.99, k_max=8)(flat, L16, z16)
assert k.item() == 8 and (w > 0).sum().item() == 8
pr = F.softmax(torch.randn(2, 500, 16) * 2.0, -1); z16 = zof(L16, torch.randn(2, 500, 32))
_, k = CAREGate(0.8)(pr, L16, z16)
print("T10 k_max strict: ok | k mean %.2f std %.2f min %d max %d (must vary, within [1,8])" % (k.float().mean(), k.float().std(), k.min(), k.max()))
assert k.float().std() > 0.3 and k.min() >= 1 and k.max() <= 8

# T11: plugs into MoELoRALinear.forward (gate swap, stats)
st, L = make_layer(N=16); st.gate = make_gate("care", tau=0.8, record=True); st.collect_stats = True
xx = torch.randn(2, 9, 32); out = L(xx)
assert out.shape == (2, 9, 64) and torch.isfinite(out).all()
h = st.stats[0][1]; assert h[0] == 0 and h[9:].sum() == 0 and h.sum() == 18
print("T11 gate swap inside layer, k histogram within [1,8]: ok")
print("\nALL GATE TESTS PASSED")
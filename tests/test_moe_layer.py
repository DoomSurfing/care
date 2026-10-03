import torch, torch.nn as nn, torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM
from moe_lora import (MoEState, FixedTopKGate, MoELoRALinear, inject_moe_lora,
                      save_moe_checkpoint, load_moe_checkpoint)
torch.manual_seed(0)

def make(N=6, r=4, d_in=32, d_out=64, drop=0.0, k=2):
    st = MoEState(FixedTopKGate(k))
    base = nn.Linear(d_in, d_out, bias=False)
    return st, base, MoELoRALinear(base, st, N, r, alpha=2 * r, dropout=drop, name="t")

# T1: zero-B init == base output
st, base, L = make(); x = torch.randn(3, 5, 32)
assert torch.allclose(L(x), base(x)); print("T1 zero-init identical to base: ok")

# T2: efficient formulation == naive Eq.1 loop
with torch.no_grad(): L.B.normal_(0, 0.1)
L.eval(); out = L(x)
probs = F.softmax(F.linear(x, L.router), -1); w, k = st.gate(probs, L, None)
naive = base(x)
for i in range(L.N):
    e = (x @ L.A[i].T) @ L.B[i].T
    naive = naive + L.scale * w[..., i:i + 1] * e
assert torch.allclose(out, naive, atol=1e-5); print("T2 matches naive dense Eq.1: ok")

# T3: gate sparsity / normalization
assert ((w > 0).sum(-1) == 2).all() and torch.allclose(w.sum(-1), torch.ones(3, 5)) and (k == 2).all()
print("T3 top-k count, weights sum to 1: ok")

# T4: gradients reach A, B, router
L.train(); L.zero_grad(); L(x).pow(2).sum().backward()
for n_, p in [("A", L.A), ("B", L.B), ("router", L.router)]:
    assert p.grad is not None and p.grad.abs().sum() > 0, n_
print("T4 grads flow to A/B/router: ok")

# T5: dropout active in train, inert in eval
_, _, D = make(drop=0.5)
with torch.no_grad(): D.B.normal_(0, 0.1)
D.train(); assert not torch.allclose(D(x), D(x))
D.eval();  assert torch.allclose(D(x), D(x)); print("T5 dropout train/eval: ok")

# T6: aux loss
st.collect_aux = True; L.train(); st.reset(); L(x)
a = st.aux_loss(); assert torch.isfinite(a) and a > 0; print(f"T6 aux loss {a.item():.3f}: ok")

# T7: tiny real Qwen2 architecture: injection, trainables, backward, save/load round-trip
def tiny():
    torch.manual_seed(1)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=200, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64))
m1 = tiny(); s1 = MoEState(FixedTopKGate(2))
assert inject_moe_lora(m1, s1, N=4, r=2, alpha=4, dropout=0.0) == 6
tr = [n for n, p in m1.named_parameters() if p.requires_grad]
assert len(tr) == 18 and all(n.endswith((".A", ".B", ".router")) for n in tr)
for mod in m1.modules():
    if isinstance(mod, MoELoRALinear): mod.B.data.normal_(0, 0.1)
ids = torch.randint(0, 200, (2, 10)); s1.collect_aux = True
m1.train(); out = m1(input_ids=ids, labels=ids); (out.loss + 0.01 * s1.aux_loss()).backward()
assert all(p.grad is not None for n, p in m1.named_parameters() if p.requires_grad)
assert all(p.grad is None for n, p in m1.named_parameters() if not p.requires_grad)
save_moe_checkpoint(m1, "/tmp/moe_t.pt")
m2 = tiny(); inject_moe_lora(m2, MoEState(FixedTopKGate(2)), N=4, r=2, alpha=4, dropout=0.0)
load_moe_checkpoint(m2, "/tmp/moe_t.pt")
m1.eval(); m2.eval()
assert torch.allclose(m1(input_ids=ids).logits, m2(input_ids=ids).logits, atol=1e-5)
ck = torch.load("/tmp/moe_t.pt"); ck["state"]["bogus.A"] = torch.zeros(1); torch.save(ck, "/tmp/moe_bad.pt")
try: load_moe_checkpoint(m2, "/tmp/moe_bad.pt"); raise SystemExit("bad ckpt NOT caught")
except AssertionError: print("T7 checkpoint round-trip + unexpected-key guard: ok")
print("\nALL MOE-LORA TESTS PASSED")
import torch
from transformers import AutoTokenizer
from common import MODEL_NAME
from data import SFTDataset, DynamicPadCollate

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
assert tok.pad_token_id is not None, "no pad token"
ds = SFTDataset("data/train/commonsense_170k.json", tok, max_len=256, limit=3000, seed=0)
print("n:", len(ds), "| left-truncated:", ds.n_trunc, "| eos:", tok.eos_token_id, "| pad:", tok.pad_token_id)

ids, lab = ds[0]
print("FULL TEXT :", repr(tok.decode(ids)))
print("SUPERVISED:", repr(tok.decode([t for t, l in zip(ids, lab) if l != -100])))

for ids, lab in ds.items:
    assert len(ids) == len(lab) <= 256
    n_sup = sum(l != -100 for l in lab)
    assert n_sup >= 2                                   # >=1 answer token + EOS
    assert all(l == -100 for l in lab[:-n_sup])         # prompt fully masked
    assert lab[-n_sup:] == ids[-n_sup:] and lab[-1] == tok.eos_token_id

batch = DynamicPadCollate(tok.pad_token_id)([ds[i] for i in range(16)])
L = batch["input_ids"].shape[1]
assert L == max(len(ds[i][0]) for i in range(16))
assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
print("batch shape:", tuple(batch["input_ids"].shape), "(padded to batch max, not 256)")
print("DATA TESTS PASSED")
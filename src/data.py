import json, random
import torch
from torch.utils.data import Dataset
from prompts import build_prompt

def split_holdout(path, n_holdout, split_seed):
    """Deterministic split. Independent of the training seed. Returns (train_list, val_list)."""
    data = json.load(open(path))
    if n_holdout <= 0:
        return data, []
    idx = list(range(len(data)))
    random.Random(split_seed).shuffle(idx)
    hold = set(idx[:n_holdout])
    train = [data[i] for i in range(len(data)) if i not in hold]
    val = [data[i] for i in idx[:n_holdout]]
    return train, val

class SFTDataset(Dataset):
    """Tokenizes prompt + bare `answer` + EOS. Over-length examples are LEFT-truncated
    on the prompt so the '### Response:' tail and the answer are always kept."""
    def __init__(self, path, tokenizer, max_len=256, train_on_inputs=False, limit=None, seed=0, examples=None):
        data = examples if examples is not None else json.load(open(path))
        if limit is not None and limit < len(data):
            data = random.Random(seed).sample(data, limit)
        eos = tokenizer.eos_token_id
        self.items, self.n_trunc = [], 0
        for ex in data:
            p = tokenizer(build_prompt(ex["instruction"], ex.get("input", "")),
                          add_special_tokens=False).input_ids
            a = tokenizer(ex["answer"], add_special_tokens=False).input_ids + [eos]
            budget = max_len - len(a)
            if len(p) > budget:
                p = p[-budget:]
                self.n_trunc += 1
            labels = (p if train_on_inputs else [-100] * len(p)) + a
            self.items.append((p + a, labels))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class DynamicPadCollate:
    """Right-pad to the longest sequence IN THE BATCH (not to max_len)."""
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, batch):
        L = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            lab[i, :len(y)] = torch.tensor(y)
            att[i, :len(x)] = 1
        return dict(input_ids=ids, attention_mask=att, labels=lab)
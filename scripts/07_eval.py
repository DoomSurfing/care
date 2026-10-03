import argparse, json, os, random, time
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MODEL_NAME, PAPER
from prompts import TASKS, build_prompt, parse_choices
from moe_lora import MoEState, MoELoRALinear, inject_moe_lora, load_moe_checkpoint
from gates import make_gate

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="checkpoint run name under ckpt/")
ap.add_argument("--ckpt", default="final")
ap.add_argument("--gate", default="fixed", choices=["fixed", "care"])
ap.add_argument("--k", type=int, default=PAPER["train_k"])
ap.add_argument("--tau", type=float, default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--tasks", nargs="*", default=list(TASKS))
ap.add_argument("--limit", type=int, default=None, help="smoke: random subset per task (seeded)")
ap.add_argument("--skip_ll", action="store_true"); ap.add_argument("--skip_gen", action="store_true")
ap.add_argument("--ll_tokens", type=int, default=12288)
ap.add_argument("--gen_bs", type=int, default=32)
ap.add_argument("--collect_k", action="store_true", help="report mean active experts per task (LL pass)")
args = ap.parse_args()
if args.gate == "care": assert args.tau is not None, "--tau required for care"
tag = args.tag or (f"fixed{args.k}" if args.gate == "fixed" else f"care_tau{args.tau:.4f}")
outdir = f"results/{args.run}"; os.makedirs(outdir, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
pad_id = tok.pad_token_id
ck_path = f"ckpt/{args.run}/{args.ckpt}.pt"
meta = torch.load(ck_path, map_location="cpu")["extra"]["config"]
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
gate = make_gate(args.gate, k=args.k, tau=args.tau)
state = MoEState(gate)
inject_moe_lora(model, state, N=meta["N"], r=meta["r"], alpha=meta["alpha"], dropout=meta["dropout"])
load_moe_checkpoint(model, ck_path)                      # strict=False + loud asserts
bn = [m.B.detach().abs().mean().item() for m in model.modules() if isinstance(m, MoELoRALinear)]
assert min(bn) > 0, "some expert B matrices are exactly zero -> untrained/mismatched checkpoint"
print(f"[ckpt] mean|B| over modules: min {min(bn):.2e} max {max(bn):.2e} | gate {tag} | trained seed {meta['seed']}")
model.eval(); state.collect_aux = False

@torch.no_grad()
def score_ll(pids, chs):
    seqs = []
    for i, (p, cs) in enumerate(zip(pids, chs)):
        for j, c in enumerate(cs):
            cid = tok(c, add_special_tokens=False).input_ids
            seqs.append((i, j, p + cid, len(p), len(cid)))
    scores = [[None] * len(cs) for cs in chs]
    khist = torch.zeros(meta["N"] + 1, dtype=torch.long)
    order = sorted(range(len(seqs)), key=lambda s: len(seqs[s][2]))
    def flush(batch):
        nonlocal khist
        L = max(len(seqs[s][2]) for s in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long); att = torch.zeros((len(batch), L), dtype=torch.long)
        for r, s in enumerate(batch):
            x = seqs[s][2]; ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
        ids, att = ids.cuda(), att.cuda()
        state.reset(); state.token_mask = att.bool(); state.collect_stats = args.collect_k
        h = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
        if args.collect_k:
            for _, hh in state.stats: khist += hh
        state.reset()
        for r, s in enumerate(batch):
            i, j, x, start, n = seqs[s]
            lp = F.log_softmax(model.lm_head(h[r, start - 1:start - 1 + n]).float(), -1)
            tgt = torch.tensor(x[start:start + n], device=lp.device)
            scores[i][j] = lp.gather(-1, tgt[:, None]).mean().item()
    batch = []
    for s in order:
        if batch and (len(batch) + 1) * len(seqs[s][2]) > args.ll_tokens:
            flush(batch); batch = []
        batch.append(s)
    if batch: flush(batch)
    state.collect_stats = False
    return [max(range(len(sc)), key=lambda j: sc[j]) for sc in scores], khist

def first_choice(text, cs):
    best = None
    for j, c in enumerate(cs):
        m = text.find(c)
        if m >= 0 and (best is None or m < best[0] or (m == best[0] and len(c) > len(cs[best[1]]))):
            best = (m, j)
    return -1 if best is None else best[1]

@torch.no_grad()
def score_gen(pids, chs):
    n = len(pids); order = sorted(range(n), key=lambda i: len(pids[i])); preds = [None] * n
    state.token_mask = None; state.collect_stats = False
    for b in range(0, n, args.gen_bs):
        sel = order[b:b + args.gen_bs]; L = max(len(pids[i]) for i in sel)
        ids = torch.full((len(sel), L), pad_id, dtype=torch.long); att = torch.zeros((len(sel), L), dtype=torch.long)
        for r, i in enumerate(sel):                                   # LEFT pad for generation
            ids[r, L - len(pids[i]):] = torch.tensor(pids[i]); att[r, L - len(pids[i]):] = 1
        out = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=8,
                             do_sample=False, pad_token_id=pad_id)
        for r, txt in enumerate(tok.batch_decode(out[:, L:], skip_special_tokens=True)):
            preds[sel[r]] = first_choice(txt, chs[sel[r]])
    return preds

results, allpreds = {}, {}
for task in args.tasks:
    t0 = time.time()
    data = json.load(open(f"data/test/{TASKS[task]}.json"))
    if args.limit and args.limit < len(data):
        keep = sorted(random.Random(0).sample(range(len(data)), args.limit)); data = [data[i] for i in keep]
    chs = [parse_choices(ex["instruction"]) for ex in data]
    assert all(c is not None for c in chs), f"{task}: choice parse failure"
    assert all(ex["answer"] in c for ex, c in zip(data, chs)), f"{task}: gold not in parsed choices"
    gold = [c.index(ex["answer"]) for ex, c in zip(data, chs)]
    pids = [tok(build_prompt(ex["instruction"], ex.get("input", "")), add_special_tokens=False).input_ids for ex in data]
    assert max(len(p) for p in pids) + 8 <= 1024, "prompt too long"
    rec, pr = dict(n=len(data), max_prompt_tok=max(len(p) for p in pids)), dict(gold=gold)
    if not args.skip_ll:
        ll, kh = score_ll(pids, chs); pr["ll"] = ll
        rec["acc_ll"] = 100 * sum(int(a == g) for a, g in zip(ll, gold)) / len(gold)
        if args.collect_k:
            rec["mean_k"] = float((kh * torch.arange(len(kh))).sum() / kh.sum()); rec["k_hist"] = kh.tolist()
    if not args.skip_gen:
        gen = score_gen(pids, chs); pr["gen"] = gen
        rec["acc_gen"] = 100 * sum(int(a == g) for a, g in zip(gen, gold)) / len(gold)
        rec["gen_unparsed"] = sum(int(a == -1) for a in gen)
    if "ll" in pr and "gen" in pr:
        rec["agree_ll_gen"] = 100 * sum(int(a == b) for a, b in zip(pr["ll"], pr["gen"])) / len(gold)
    rec["sec"] = time.time() - t0
    results[task], allpreds[task] = rec, pr
    print(f"[{task}] " + " | ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items() if k != "k_hist"), flush=True)

summ = {m: sum(r[m] for r in results.values()) / len(results) for m in ("acc_ll", "acc_gen") if all(m in r for r in results.values())}
print("AVERAGE (macro over tasks):", {k: round(v, 2) for k, v in summ.items()})
json.dump(dict(run=args.run, tag=tag, gate=args.gate, k=args.k, tau=args.tau, limit=args.limit, seed=meta["seed"],
               tasks=results, **{"avg_" + m[4:]: v for m, v in summ.items()}), open(f"{outdir}/{tag}.json", "w"), indent=1)
json.dump(allpreds, open(f"{outdir}/{tag}.preds.json", "w"))
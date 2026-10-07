import json, os, statistics as st

def load(path):
    if not path or not os.path.exists(path): return []
    return [json.loads(l) for l in open(path) if l.strip()]

def find(run, suffix):
    for p in (f"logs/train/{run}.{suffix}", f"results/{run}/train_logs/{run}.{suffix}", f"results/guard30_logs/{run}.{suffix}"):
        if os.path.exists(p): return p
    return None

def dedupe(rs):
    d = {}
    for r in rs: d[r["step"]] = r
    return [d[k] for k in sorted(d)]

EPOCH3 = 2 * 10526

for s in (42, 43, 44, 45, 46):
    run = f"moe_fixed4_s{s}"
    main = dedupe(load(find(run, "jsonl")))
    if not main:
        print(f"\ns{s}: no log found"); continue
    last = main[-1]
    sk_total = last.get("skipped", 0)
    print(f"\n=== s{s} | last step {last['step']} | skipped {sk_total} ({100 * sk_total / last['step']:.1f}%)")
    out = []
    for w in range(0, last["step"], 3000):
        seg = [r for r in main if w <= r["step"] <= w + 3000]
        if len(seg) >= 2 and seg[-1]["step"] > seg[0]["step"]:
            out.append(f"{w // 1000}k:{100 * (seg[-1]['skipped'] - seg[0]['skipped']) / (seg[-1]['step'] - seg[0]['step']):.0f}%")
    print("  skip rate per 3k-step window:", " ".join(out))
    
    sk = load(find(run, "skips.jsonl"))
    if sk:
        g = [r["gn_preclip"] for r in sk]
        print(f"  skipped steps with gn<10: {100 * sum(x < 10 for x in g) / len(g):.0f}% | gn<50: {100 * sum(x < 50 for x in g) / len(g):.0f}% | median skipped gn {st.median(g):.1f}")
        late = [r["median"] for r in sk if r["step"] > EPOCH3 and r.get("median")]
        if late:
            print(f"  epoch 3: {len(late)} skips | typical rolling median {st.median(late):.4f} -> cutoff ~{30 * st.median(late):.2f}")
    
    va = dedupe(load(find(run, "val.jsonl")))
    if va:
        pick = va[::5] + ([va[-1]] if (len(va) - 1) % 5 else [])
        print("  val step:loss/exact ->", " ".join(f"{r['step'] // 1000}k:{r['val_loss']:.3f}/{r['val_exact']:.1f}" for r in pick))

old = dedupe(load("results/_superseded_noguard/moe_fixed4_s43/train_logs/moe_fixed4_s43.val.jsonl"))
if old:
    pick = old[::5] + ([old[-1]] if (len(old) - 1) % 5 else [])
    print("\n=== OLD s43 (guard OFF, archived) val ->", " ".join(f"{r['step'] // 1000}k:{r['val_loss']:.3f}/{r['val_exact']:.1f}" for r in pick))

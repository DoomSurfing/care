import argparse, json, statistics as st
ap = argparse.ArgumentParser()
ap.add_argument("--method", default="moe_fixed4"); ap.add_argument("--tag", default="fixed4")
ap.add_argument("--seeds", nargs="+", type=int, required=True)
args = ap.parse_args()
R = [json.load(open(f"results/{args.method}_s{s}/{args.tag}.json")) for s in args.seeds]
for m in ("ll", "gen"):
    if f"avg_{m}" not in R[0]: continue
    print(f"\n== {args.method} / {args.tag} / {m.upper()} scoring / seeds {args.seeds} ==")
    for t in R[0]["tasks"]:
        v = [r["tasks"][t][f"acc_{m}"] for r in R]
        print(f"{t:14} {st.mean(v):6.2f} ± {st.stdev(v) if len(v) > 1 else 0:4.2f}   {[round(x, 2) for x in v]}")
    a = [r[f"avg_{m}"] for r in R]
    print(f"{'AVERAGE':14} {st.mean(a):6.2f} ± {st.stdev(a) if len(a) > 1 else 0:4.2f}   (sample std, ddof=1)")
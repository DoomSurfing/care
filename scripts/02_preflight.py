import json, sys, argparse
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
from common import MODEL_NAME
from prompts import TASKS, build_prompt, parse_choices

ap = argparse.ArgumentParser()
ap.add_argument("--show", action="store_true", help="print one real example per task")
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def check(name, data):
    parse_fail = gold_bad = 0
    prompts, answers, csets = [], Counter(), Counter()
    for ex in data:
        ch = parse_choices(ex["instruction"])
        if ch is None:
            parse_fail += 1
        elif ex["answer"] not in ch:
            gold_bad += 1
        else:
            csets["/".join(ch)] += 1
        prompts.append(build_prompt(ex["instruction"], ex.get("input", "")))
        answers[ex["answer"]] += 1
    plen = np.array([len(x) for x in tok(prompts, add_special_tokens=False).input_ids])
    alen = np.array([len(x) for x in tok([ex["answer"] for ex in data], add_special_tokens=False).input_ids])
    full = plen + alen + 1  # + EOS
    row = dict(n=len(data), parse_fail=parse_fail, gold_not_in_choices=gold_bad,
               avg=round(float(plen.mean()), 1), p95=float(np.percentile(plen, 95)), max=int(plen.max()),
               max_answer_tok=int(alen.max()), full_gt256=int((full > 256).sum()))
    print(f"\n[{name}] {row}")
    print("  choice sets:", csets.most_common(3))
    print("  answer dist:", answers.most_common(6))
    return row, prompts

results, ok = {}, True
for task, stem in TASKS.items():
    data = json.load(open(f"data/test/{stem}.json"))
    row, prompts = check(task, data)
    results[task] = row
    ok &= (row["parse_fail"] == 0 and row["gold_not_in_choices"] == 0)
    if args.show:
        ex = data[0]
        print("  --- example prompt repr:\n ", repr(prompts[0]))
        print("  gold:", repr(ex["answer"]), "| parsed choices:", parse_choices(ex["instruction"]))

train = json.load(open("data/train/commonsense_170k.json"))
row, _ = check("TRAIN(all)", train)
results["train"] = row
ok &= (row["parse_fail"] == 0 and row["gold_not_in_choices"] == 0)

json.dump(results, open("results/preflight.json", "w"), indent=1)
print("\nPREFLIGHT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
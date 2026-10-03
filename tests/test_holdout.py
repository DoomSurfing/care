from data import split_holdout
P = "data/train/commonsense_170k.json"
tr, va = split_holdout(P, 2000, 1234)
tr2, va2 = split_holdout(P, 2000, 1234)
tr3, va3 = split_holdout(P, 2000, 999)
assert len(va) == 2000 and len(tr) + len(va) == 170420, (len(tr), len(va))
assert va == va2 and tr == tr2, "split not deterministic"
assert va != va3, "split seed has no effect"
print("train", len(tr), "| val", len(va), "| steps/epoch @bs16:", len(tr) // 16)
print("HOLDOUT TESTS PASSED")
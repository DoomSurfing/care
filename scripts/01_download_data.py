import json, os, urllib.request

BASE = "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main"
TRAIN_URL = f"{BASE}/ft-training_set/commonsense_170k.json"
TEST_DIRS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande",
             "ARC-Challenge", "ARC-Easy", "openbookqa"]

def fetch(url, dst):
    if os.path.exists(dst):
        print("exists:", dst); return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(url, dst)
    print("downloaded:", dst)

fetch(TRAIN_URL, "data/train/commonsense_170k.json")
for d in TEST_DIRS:
    fetch(f"{BASE}/dataset/{d}/test.json", f"data/test/{d}.json")

tr = json.load(open("data/train/commonsense_170k.json"))
print("\ntrain:", len(tr), "keys:", sorted(tr[0].keys()))
for d in TEST_DIRS:
    data = json.load(open(f"data/test/{d}.json"))
    print(f"test {d:14} n={len(data):6} keys={sorted(data[0].keys())}")
    
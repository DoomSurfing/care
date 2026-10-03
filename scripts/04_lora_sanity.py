import argparse, time, json, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from common import MODEL_NAME, PAPER, ASSUMPTIONS, print_config
from data import SFTDataset, DynamicPadCollate
from prompts import build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--bs", type=int, default=4)
args = ap.parse_args()
print_config()
torch.manual_seed(ASSUMPTIONS["seed"])

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").cuda()
model.config.use_cache = False
cfg = LoraConfig(r=PAPER["r"], lora_alpha=ASSUMPTIONS["lora_alpha"], lora_dropout=PAPER["dropout"],
                 target_modules=["gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()

ds = SFTDataset("data/train/commonsense_170k.json", tok, max_len=ASSUMPTIONS["max_len_train"],
                train_on_inputs=ASSUMPTIONS["train_on_inputs"], limit=args.steps * args.bs, seed=0)
dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=DynamicPadCollate(tok.pad_token_id))
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=PAPER["lr"], weight_decay=ASSUMPTIONS["weight_decay"])
sched = get_cosine_schedule_with_warmup(opt, ASSUMPTIONS["warmup_steps"], args.steps)

model.train()
losses, t0 = [], time.time()
torch.cuda.reset_peak_memory_stats()
for step, b in enumerate(dl):
    b = {k: v.cuda() for k, v in b.items()}
    loss = model(**b).loss
    loss.backward()
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    losses.append(loss.item())
    if step % 5 == 0:
        print(f"step {step:3d} loss {loss.item():.4f} seqlen {b['input_ids'].shape[1]} "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    if step + 1 >= args.steps: break

f10, l10 = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
print(f"\nmean first10 {f10:.4f} -> last10 {l10:.4f} | {time.time()-t0:.0f}s")
print("LOSS DECREASED:", l10 < f10)

# generation sanity (format only; 60 steps won't give accuracy)
model.eval(); model.config.use_cache = True
for stem in ["boolq", "piqa"]:
    ex = json.load(open(f"data/test/{stem}.json"))[0]
    inp = tok(build_prompt(ex["instruction"], ex.get("input", "")), return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
    print(stem, "| gold:", ex["answer"], "| gen:", repr(tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)))
import argparse, glob, json, math, os, random, subprocess, sys, time
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MODEL_NAME, PAPER, ASSUMPTIONS, print_config
from data import SFTDataset, DynamicPadCollate, split_holdout
from moe_lora import MoEState, FixedTopKGate, inject_moe_lora, save_moe_checkpoint, load_moe_checkpoint
from stability import SpikeGuard, LossWatchdog

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="moe_fixed4", choices=["moe_fixed4"])
ap.add_argument("--seed", type=int, default=ASSUMPTIONS["seed"])
ap.add_argument("--run_name", default=None)
ap.add_argument("--max_steps", type=int, default=None, help="SMOKE ONLY: shortens the run (cosine horizon and warmup shrink too)")
ap.add_argument("--stop_at", type=int, default=None, help="save last.pt and exit at this step; the full-run schedule is kept, so it can be resumed")
ap.add_argument("--save_every", type=int, default=1000)
ap.add_argument("--snap_every", type=int, default=2000)
ap.add_argument("--keep_snaps", type=int, default=3)
ap.add_argument("--log_every", type=int, default=10)
ap.add_argument("--val_every", type=int, default=1000, help="held-out loss every N steps (0 = off)")
ap.add_argument("--beta2", type=float, default=ASSUMPTIONS["adam_beta2"])
ap.add_argument("--skip_mult", type=float, default=ASSUMPTIONS["spike_skip_mult"], help="0 = only non-finite steps are skipped")
ap.add_argument("--no_watchdog", action="store_true")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--wandb", action="store_true")
ap.add_argument("--wandb_project", default="hcare-moe-lora")
ap.add_argument("--aux_coef", type=float, default=ASSUMPTIONS["aux_coef"])
ap.add_argument("--val_holdout", type=int, default=ASSUMPTIONS["val_holdout"])
args = ap.parse_args()

run = args.run_name or f"{args.method}_s{args.seed}"
ckdir = f"ckpt/{run}"; os.makedirs(ckdir, exist_ok=True); os.makedirs("logs/train", exist_ok=True)
final_path, last_path = f"{ckdir}/final.pt", f"{ckdir}/last.pt"
if os.path.exists(final_path) and args.max_steps is None:
    print(f"[skip] {final_path} already exists"); sys.exit(0)

def git_info():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=10).stdout.strip()
        return dict(commit=h or None, dirty=bool(d))
    except Exception:
        return dict(commit=None, dirty=None)
GIT = git_info()

print(f"=== RUN {run} | seed {args.seed} | git {GIT} ===")
if GIT["dirty"]:
    print("[WARNING] uncommitted changes in tracked files: commit before real runs so the hash identifies the code")
print_config()
print("args:", vars(args))
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
train_ex, val_ex = split_holdout("data/train/commonsense_170k.json", args.val_holdout, ASSUMPTIONS["val_split_seed"])
if val_ex and not os.path.exists("data/val_holdout.json"):
    json.dump(val_ex, open("data/val_holdout.json", "w"))
ds = SFTDataset(None, tok, max_len=ASSUMPTIONS["max_len_train"], train_on_inputs=ASSUMPTIONS["train_on_inputs"], examples=train_ex)
bs = PAPER["batch_size"]
steps_per_epoch = len(ds) // bs                       # drop_last
total_steps = args.max_steps or steps_per_epoch * PAPER["epochs"]
warmup_steps = int(ASSUMPTIONS["warmup_frac"] * total_steps)
print(f"train n={len(ds)} | left-truncated {ds.n_trunc} | steps/epoch {steps_per_epoch} | total steps {total_steps} | "
      f"warmup {warmup_steps} steps ({ASSUMPTIONS['warmup_frac']:.0%}) | AdamW betas (0.9, {args.beta2}) | clip {ASSUMPTIONS['grad_clip']}")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.config.use_cache = False
state = MoEState(FixedTopKGate(PAPER["train_k"]))
n_inj = inject_moe_lora(model, state, N=PAPER["N"], r=PAPER["r"], alpha=ASSUMPTIONS["lora_alpha"],
                        dropout=PAPER["dropout"], router_std=ASSUMPTIONS["router_init_std"])
params = [p for p in model.parameters() if p.requires_grad]
print(f"injection points {n_inj} | trainable {sum(p.numel() for p in params)/1e6:.1f}M | "
      f"grad checkpointing: {getattr(model, 'is_gradient_checkpointing', False)} | physical batch {bs} (no accumulation)")
opt = torch.optim.AdamW(params, lr=PAPER["lr"], betas=(0.9, args.beta2), weight_decay=ASSUMPTIONS["weight_decay"])
state.collect_aux = True

def lr_at(step):
    w = warmup_steps
    if w and step < w:
        return PAPER["lr"] * (step + 1) / w
    prog = (step - w) / max(1, total_steps - w)
    return PAPER["lr"] * 0.5 * (1 + math.cos(math.pi * prog))

def epoch_perm(epoch):
    g = torch.Generator().manual_seed(args.seed * 100003 + epoch)
    return torch.randperm(len(ds), generator=g).tolist()

config = dict(N=PAPER["N"], r=PAPER["r"], alpha=ASSUMPTIONS["lora_alpha"], dropout=PAPER["dropout"],
              train_k=PAPER["train_k"], aux_coef=args.aux_coef, seed=args.seed, total_steps=total_steps,
              method=args.method, router_std=ASSUMPTIONS["router_init_std"], lr=PAPER["lr"],
              weight_decay=ASSUMPTIONS["weight_decay"], grad_clip=ASSUMPTIONS["grad_clip"],
              warmup_frac=ASSUMPTIONS["warmup_frac"], warmup_steps=warmup_steps,
              beta2=args.beta2, skip_mult=args.skip_mult)

start = 0
if args.resume and os.path.exists(last_path):
    extra = load_moe_checkpoint(model, last_path)
    assert extra["config"] == config, f"config mismatch on resume: {extra['config']} vs {config}"
    opt.load_state_dict(extra["opt"]); start = extra["step"]
    torch.set_rng_state(extra["rng_cpu"]); torch.cuda.set_rng_state(extra["rng_cuda"])
    print(f"[resume] continuing from step {start}")
if args.stop_at and start >= args.stop_at:
    print(f"[stop_at] start step {start} >= stop_at {args.stop_at}: nothing to do"); sys.exit(0)

def save_last(step):
    tmp = last_path + ".tmp"
    save_moe_checkpoint(model, tmp, extra=dict(config=config, opt=opt.state_dict(), step=step,
                        rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()))
    os.replace(tmp, last_path)

def rotate_snaps():
    snaps = sorted(glob.glob(f"{ckdir}/snap_*.pt"), key=lambda f: int(os.path.basename(f)[5:-3]))
    for f in snaps[:-args.keep_snaps]: os.remove(f)

if args.wandb:
    try:
        import wandb
        _wf = f"{ckdir}/wandb_id.txt"
        wid = open(_wf).read().strip() if os.path.exists(_wf) else f"{run}-{int(time.time())}"
        open(_wf, "w").write(wid)
        wandb.init(project=args.wandb_project, name=run, id=wid, resume="allow",
                   config=dict(paper=PAPER, assumptions=ASSUMPTIONS, args=vars(args), git=GIT))
    except Exception as e:
        print(f"[WARNING] W&B disabled ({type(e).__name__}: {str(e)[:120]}). Training continues; logs/train/{run}.jsonl is still written.")
        args.wandb = False

collate = DynamicPadCollate(tok.pad_token_id)
val_items = []
if args.val_every and val_ex:
    _vds = SFTDataset(None, tok, max_len=ASSUMPTIONS["max_len_train"], train_on_inputs=False, examples=val_ex)
    val_items = sorted(_vds.items, key=lambda x: len(x[0]))

@torch.no_grad()
def run_val():
    """Held-out loss on answer+EOS tokens and exact-match of the full answer sequence."""
    model.eval(); state.collect_aux = False
    loss_sum, ntok, ex_ok = 0.0, 0, 0
    for i in range(0, len(val_items), 32):
        b = {k: v.cuda() for k, v in collate(val_items[i:i + 32]).items()}
        state.reset(); state.token_mask = b["attention_mask"].bool()
        h = model.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).last_hidden_state
        lab = b["labels"][:, 1:]; sel = lab != -100
        logits = model.lm_head(h[:, :-1][sel]).float(); tgt = lab[sel]
        loss_sum += F.cross_entropy(logits, tgt, reduction="sum").item()
        ok = logits.argmax(-1) == tgt
        ntok += tgt.numel()
        wrong = torch.zeros(lab.shape[0], device=lab.device).index_add_(0, sel.nonzero()[:, 0], (~ok).float())
        ex_ok += int((wrong == 0).sum().item())
    state.reset(); model.train(); state.collect_aux = True
    return loss_sum / ntok, 100.0 * ex_ok / len(val_items)

logf = open(f"logs/train/{run}.jsonl", "a")
spikef = open(f"logs/train/{run}.spikes.jsonl", "a")
skipf = open(f"logs/train/{run}.skips.jsonl", "a")
valf = open(f"logs/train/{run}.val.jsonl", "a")
guard = SpikeGuard(mult=args.skip_mult)
wd = None if args.no_watchdog else LossWatchdog()
model.train(); torch.cuda.reset_peak_memory_stats()
t0, perm_cache, acc = time.time(), {}, dict(loss=0.0, aux=0.0, n=0)
for step in range(start, total_steps):
    epoch, i = divmod(step, steps_per_epoch)
    if epoch not in perm_cache:
        perm_cache = {epoch: epoch_perm(epoch)}
    idx = perm_cache[epoch][i * bs:(i + 1) * bs]
    b = {k: v.cuda(non_blocking=True) for k, v in collate([ds[j] for j in idx]).items()}
    lr = lr_at(step)
    for g in opt.param_groups: g["lr"] = lr
    state.reset(); state.token_mask = b["attention_mask"].bool()
    out = model(**b)
    aux = state.aux_loss()
    loss = out.loss + args.aux_coef * aux
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {step}: lm={out.loss.item()} aux={float(aux.detach())}")
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, ASSUMPTIONS["grad_clip"] or float("inf"))   # inf => log only
    gnv = float(gn)
    if math.isfinite(gnv) and gnv > 5.0:   # spike attribution: share of the (post-clip) grad norm per parameter tensor
        pn = sorted(((q.grad.norm().item(), n) for n, q in model.named_parameters()
                     if q.requires_grad and q.grad is not None), reverse=True)
        tot = math.sqrt(sum(v * v for v, _ in pn)) + 1e-12
        spikef.write(json.dumps(dict(step=step + 1, gn_preclip=gnv, lm_loss=out.loss.item(),
                     aux=float(aux.detach()), seqlen=int(b["input_ids"].shape[1]), batch_idx=idx,
                     top=[(n, round(v / tot, 3)) for v, n in pn[:6]])) + "\n"); spikef.flush()
    skip, med = guard.should_skip(gnv)
    if skip:
        skipf.write(json.dumps(dict(step=step + 1, gn_preclip=gnv, median=med, lm_loss=out.loss.item())) + "\n"); skipf.flush()
    else:
        opt.step()
    opt.zero_grad(set_to_none=True)
    _l = out.loss.item()
    acc["loss"] += _l; acc["aux"] += float(aux.detach()); acc["n"] += 1
    done = step + 1
    tripped = wd is not None and wd.update(_l)
    if done % args.log_every == 0 or done == total_steps:
        el = time.time() - t0
        rec = dict(step=done, epoch=done / steps_per_epoch, loss=acc["loss"] / acc["n"], aux=acc["aux"] / acc["n"],
                   grad_norm=gnv, lr=lr, seqlen=int(b["input_ids"].shape[1]), skipped=guard.n_skipped,
                   peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                   eta_h=el / max(1, done - start) * (total_steps - done) / 3600)
        acc = dict(loss=0.0, aux=0.0, n=0)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if args.wandb: wandb.log(rec, step=done)
        print(f"step {done}/{total_steps} loss {rec['loss']:.4f} aux {rec['aux']:.3f} gn {rec['grad_norm']:.2f} "
              f"seq {rec['seqlen']} reserved {rec['peak_reserved_gib']:.1f} GiB skipped {guard.n_skipped} ETA {rec['eta_h']:.2f} h", flush=True)
    if tripped:
        p = f"{ckdir}/tripped_{done}.pt"
        save_moe_checkpoint(model, p, extra=dict(config=config, step=done))
        print(f"[WATCHDOG] step {done}: 200-step mean lm loss {wd.last_mean:.3f} vs best {wd.best:.3f}, elevated for {wd.patience}+ steps. "
              f"Saved {p}. Last good state: {last_path} / snap_*.pt. Exiting with code 2.", flush=True)
        sys.exit(2)
    if done % args.save_every == 0 and done < total_steps:
        save_last(done)
        if done % args.snap_every == 0:
            import shutil; shutil.copy(last_path, f"{ckdir}/snap_{done}.pt"); rotate_snaps()
    if val_items and done % args.val_every == 0:
        vl, vem = run_val()
        valf.write(json.dumps(dict(step=done, val_loss=vl, val_exact=vem)) + "\n"); valf.flush()
        if args.wandb: wandb.log(dict(val_loss=vl, val_exact=vem), step=done)
        print(f"[VAL] step {done} val_loss {vl:.4f} exact_match {vem:.2f}", flush=True)
    if args.stop_at and done == args.stop_at:
        save_last(done); print(f"[stop_at] saved last.pt at step {done}, exiting"); sys.exit(0)

save_moe_checkpoint(model, final_path, extra=dict(config=config, step=total_steps))
if args.max_steps is None:
    if os.path.exists(last_path): os.remove(last_path)
    for f in glob.glob(f"{ckdir}/snap_*.pt"): os.remove(f)
json.dump(dict(run=run, config=config, total_steps=total_steps, minutes=(time.time() - t0) / 60, n_skipped=guard.n_skipped,
               git=GIT, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30), open(f"{ckdir}/train_summary.json", "w"), indent=1)
print(f"DONE {run} -> {final_path}")
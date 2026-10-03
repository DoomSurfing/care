import argparse, time, statistics, torch
from transformers import AutoModelForCausalLM
from common import MODEL_NAME, PAPER, ASSUMPTIONS
from moe_lora import MoEState, FixedTopKGate, inject_moe_lora

ap = argparse.ArgumentParser()
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--micro", type=int, default=16, help="physical micro-batch (bs must be divisible)")
ap.add_argument("--seq", type=int, default=256)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--ckpt", action="store_true")
args = ap.parse_args()
assert args.bs % args.micro == 0
tag = f"bs{args.bs} micro{args.micro} seq{args.seq} ckpt={args.ckpt}"

def is_oom(e):
    s = str(e)
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in s.lower() or "NVML_SUCCESS" in s

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").cuda()
model.config.use_cache = False
state = MoEState(FixedTopKGate(PAPER["train_k"]))
inject_moe_lora(model, state, N=PAPER["N"], r=PAPER["r"], alpha=ASSUMPTIONS["lora_alpha"], dropout=PAPER["dropout"])
params = [p for p in model.parameters() if p.requires_grad]
print(f"[{tag}] trainable params: {sum(p.numel() for p in params)/1e6:.1f}M | after load: {torch.cuda.memory_allocated()/2**30:.1f} GiB")
if args.ckpt:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()
use_aux = not args.ckpt
state.collect_aux = use_aux
opt = torch.optim.AdamW(params, lr=PAPER["lr"], weight_decay=ASSUMPTIONS["weight_decay"])

ids_full = torch.randint(1000, 100000, (args.bs, args.seq), device="cuda")
torch.cuda.reset_peak_memory_stats()
times = []
try:
    for s in range(args.steps):
        torch.cuda.synchronize(); t0 = time.time()
        n_micro = args.bs // args.micro
        for chunk in ids_full.chunk(n_micro):
            state.reset(); state.token_mask = torch.ones_like(chunk, dtype=torch.bool)
            out = model(input_ids=chunk, attention_mask=torch.ones_like(chunk), labels=chunk)
            loss = out.loss + (ASSUMPTIONS["aux_coef"] * state.aux_loss() if use_aux else 0.0)
            (loss / n_micro).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        print(f"[{tag}] step {s} {times[-1]:.2f}s loss {loss.item():.3f}")
    med = statistics.median(times[1:] or times)
    steps_total = 170_000 * PAPER["epochs"] / PAPER["batch_size"]
    print(f"[{tag}] RESULT OK | peak alloc {torch.cuda.max_memory_allocated()/2**30:.1f} GiB | "
          f"peak reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GiB | median {med:.2f} s/step | "
          f"worst-case full-run est {med*steps_total/3600:.1f} h (real dynamic-padded batches are shorter)")
except Exception as e:
    if is_oom(e):
        print(f"[{tag}] RESULT OOM (peak alloc {torch.cuda.max_memory_allocated()/2**30:.1f} GiB): {str(e)[:120]}")
    else:
        raise
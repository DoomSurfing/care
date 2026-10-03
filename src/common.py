MODEL_NAME = "Qwen/Qwen2.5-7B"

# ---- Values given by the paper (Appendix B) ----
PAPER = dict(N=16, k_min=1, k_max=8, gamma=2, delta=0.55, B=4, w=0.5,
             r=8, lr=2e-4, batch_size=16, dropout=0.05, epochs=3, train_k=4)

# ---- ASSUMPTIONS: not specified by the paper / context doc. Change here only. ----
ASSUMPTIONS = dict(
    lora_alpha=16,              # 2 x rank, common default
    aux_coef=0.01,              # Switch-style load-balancing coefficient (mechanism is paper-mandated, value is not)
    weight_decay=0.0,           # neutral, not the AdamW default
    grad_clip=1.0,              # team-verified against the reference setup (also the HF Trainer default)
    warmup_frac=0.03,           # linear warmup = 3% of total steps (VI-MoLE Table 6, same authors); CARE does not state warmup
    warmup_steps=0,             # DEPRECATED, unused by 06_train.py (04_lora_sanity.py still reads it)
    adam_beta2=0.95,            # ASSUMPTION: Adam betas unspecified in CARE and VI-MoLE; PyTorch default 0.999 was unstable here
    spike_skip_mult=0.0,        # 0 = off (only non-finite steps are skipped). >0 skips steps with grad-norm > mult x rolling median
    max_len_train=256,          # LLM-Adapters cutoff; over-length prompts LEFT-truncated (answer always kept)
    train_on_inputs=False,      # loss on answer + EOS only
    router_per_module=True,     # one router (N x d) per injection point
    router_init_std=0.01,
    shared_dropout_mask_across_experts=True,
    aux_averaged_over_layers=True,
    router_logits_in_activation_dtype=True,   # softmax done in fp32
    bf16_frozen_base_fp32_adapter=True,       # base weights bf16, adapter params fp32 (cast to bf16 in forward)
    drop_last=True,             # 170420-2000 examples -> full batches of 16 only
    val_holdout=2000,           # held out of training for calibration / tuning (same for every method+seed)
    val_split_seed=1234,        # fixed: split does NOT change with the training seed
    ll_score="mean_logprob_over_choice_tokens",
    seed=42,                    # default only; real runs pass --seed
)

SEEDS = [42, 43, 44, 45, 46]    # default seed list; override per method with SEEDS="..." env var in the queue script

def print_config():
    print("PAPER (fixed):", PAPER)
    print("ASSUMPTIONS (labeled, not from paper):", ASSUMPTIONS)
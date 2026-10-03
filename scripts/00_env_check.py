import sys, subprocess, importlib, torch
import torch.nn.functional as F

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"ERR: {e}"

print("python      :", sys.version.split()[0])
print("torch       :", torch.__version__, "| torch built for CUDA", torch.version.cuda)
print("nvcc (sys)  :", sh("nvcc --version | tail -n 1") or "not found")
print("nvidia-smi  :", sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"))
for m in ["transformers", "datasets", "accelerate", "peft", "wandb", "numpy"]:
    try:
        print(f"{m:12}:", importlib.import_module(m).__version__)
    except Exception as e:
        print(f"{m:12}: MISSING ({e})")

assert torch.cuda.is_available(), "CUDA not available"
free, total = torch.cuda.mem_get_info()
print(f"mem_get_info: free {free/2**30:.1f} GiB / total {total/2**30:.1f} GiB")
print("bf16 supported:", torch.cuda.is_bf16_supported())

q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
print("SDPA ok:", tuple(o.shape),
      "| flash_sdp:", torch.backends.cuda.flash_sdp_enabled(),
      "| mem_efficient_sdp:", torch.backends.cuda.mem_efficient_sdp_enabled())

chunks = []
try:
    while True:
        chunks.append(torch.empty(2**30, dtype=torch.uint8, device="cuda"))
except Exception as e:
    print(f"allocated {len(chunks)} GiB before {type(e).__name__}: {str(e)[:150]}")
del chunks
torch.cuda.empty_cache()
#!/usr/bin/env python3
"""
Smoke-test the model on this machine: does it build, run, and fit in memory?

Needs NO training data — inputs are random tensors of the right shape — so it
validates the environment before any staging or download has to be finished.

WHAT IT CHECKS
--------------
1. torch sees the GPU, and which one (the Roihu login node has no nvidia-smi,
   so this is the first place the GH200's memory figure actually appears)
2. UNetModel3D builds from the real config, at the real 192x288 resolution
3. forward + backward runs and produces finite gradients
4. PEAK MEMORY per (batch, seq_len) — which is the number the monthly plan
   turns on. seq_len 1 -> 12 multiplies activations by ~12, and whether that
   needs batch_size 1 or can stay higher is a hardware question, not one that
   can be answered from the LUMI MI250X numbers.

--sweep walks seq_len until it runs out of memory, reporting the largest that
fits. That is the empirical answer to "can we train monthly here".

Usage
-----
    python scripts/smoke_test_model.py                       # default 1 frame
    python scripts/smoke_test_model.py --seq-len 12 --batch 1
    python scripts/smoke_test_model.py --sweep               # find the limit
    python scripts/smoke_test_model.py --config configs/config_aero_monthly.yaml
"""
import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))["model"]
    kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
    from models.video_net import UNetModel3D
    return UNetModel3D(**kwargs), kwargs


def try_shape(model, kw, batch, seq_len, H, W, device, dtype):
    """One forward+backward. Returns (ok, peak_GiB, seconds, message)."""
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    try:
        x = torch.randn(batch, kw["in_channels"], seq_len, H, W, device=device)
        cond = torch.randn(batch, kw["cond_channels"], seq_len, H, W, device=device)
        # The trainer calls model(x, scheduler.log_snr(t), cond_map=cond): the
        # second argument is a FLOAT log-SNR, not an integer step index, and the
        # conditioning goes in cond_map — passing it positionally puts it in
        # `days` instead and the model never sees it.
        t = torch.randn(batch, device=device)
        t0 = time.time()
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(dtype is not None)):
            out = model(x, t, cond_map=cond)
            loss = out.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda" else float("nan"))
        g = [p.grad for p in model.parameters() if p.grad is not None]
        finite = all(torch.isfinite(gi).all().item() for gi in g[:20])
        model.zero_grad(set_to_none=True)
        del x, cond, out, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return True, peak, dt, ("ok" if finite else "NON-FINITE GRADS")
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, float("nan"), 0.0, "OOM"
    except Exception as e:                      # noqa: BLE001 - report anything
        return False, float("nan"), 0.0, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config_aero.yaml")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=1)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=288)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--sweep", action="store_true",
                    help="increase seq_len until OOM, reporting the largest fit")
    args = ap.parse_args()

    print("=" * 62)
    print(f" torch {torch.__version__}   python {sys.version.split()[0]}")
    print(f" cuda available: {torch.cuda.is_available()}   "
          f"devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"   gpu{i}: {p.name}  {p.total_memory/2**30:.0f} GiB  "
                  f"sm_{p.major}{p.minor}")
    else:
        print("   NO GPU — this will run on CPU and prove only that it builds")
    print("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if args.precision == "bf16" and device.type == "cuda"
             else None)

    model, kw = build(args.config)
    n = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.config}")
    print(f"[model] in={kw['in_channels']} out={kw['out_channels']} "
          f"cond={kw['cond_channels']} dim={kw['model_dim']} "
          f"mults={kw['dim_mults']}")
    print(f"[model] {n/1e6:.1f}M parameters")
    model = model.to(device)

    seqs = ([args.seq_len] if not args.sweep
            else [1, 2, 3, 6, 12, 18, 24])
    print(f"\n{'seq_len':>8} {'batch':>6} {'peak GiB':>10} {'fwd+bwd s':>10}  status")
    print("-" * 62)
    best = 0
    for s in seqs:
        ok, peak, dt, msg = try_shape(model, kw, args.batch, s,
                                      args.height, args.width, device, dtype)
        print(f"{s:>8} {args.batch:>6} {peak:>10.2f} {dt:>10.2f}  {msg}")
        if ok and msg == "ok":
            best = s
        elif args.sweep:
            break

    print("-" * 62)
    if best:
        print(f"[result] largest seq_len that fits at batch {args.batch}: {best}")
        if args.sweep and best >= 12:
            print("[result] seq_len 12 fits — monthly training is viable at this "
                  "batch size")
        elif args.sweep:
            print(f"[result] seq_len 12 does NOT fit at batch {args.batch}; "
                  f"reduce batch or enable gradient checkpointing "
                  f"(use_checkpoint threads through video_net.py but is unset "
                  f"in every config)")
        return 0
    print("[result] nothing ran — see the status column", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

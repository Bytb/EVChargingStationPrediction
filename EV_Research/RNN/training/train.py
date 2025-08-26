# EV_Research/RNN/training/train.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import time
import yaml
import torch
import torch.optim as optim
import random, numpy as np
# --- make "from EV_Research..." work when running this file directly ---
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # parent of EV_Research
from contextlib import nullcontext
# ----------------------------------------------------------------------

from EV_Research.RNN.data.prep.Loaders import make_dataloader, EVSequenceDataset
from EV_Research.RNN.models.registry import make_model
from EV_Research.RNN.training.loss import get_loss
from EV_Research.RNN.training.eval import compute_metrics
from EV_Research.RNN.training.callbacks import EarlyStopper, CheckpointManager

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _device_from_cfg(cfg: Dict) -> torch.device:
    dev = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

def _out_dir_from_cfg(cfg: Dict) -> Path:
    city = cfg["dataset"]["city"]
    model_name = cfg["train"]["model"]
    base = cfg.get("logging", {}).get("out_dir", f"runs/{city}/{model_name}")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(base) / ts
    out.mkdir(parents=True, exist_ok=True)
    return out

def _epoch(
    dataloader,
    model,
    criterion,
    optimizer,
    device,
    *,
    amp: bool = False,
    train: bool = True,
    grad_clip: float = 1.0,
) -> float:
    """
    One pass over dataloader. Returns mean loss (masked) as float.
    - If amp=True and device is CUDA, uses torch.amp autocast + GradScaler.
    - If grad_clip > 0, applies clip_grad_norm_ in FP32 (unscales first).
    """
    model.train(mode=train)

    # Enable AMP only when requested AND on CUDA
    use_amp = bool(amp) and (getattr(device, "type", str(device)) == "cuda")
    scaler = torch.amp.GradScaler("cuda") if (train and use_amp) else None
    autocast_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        X   = batch["X_seq"].to(device)        # [B,L,N,F]
        A = batch["A"].to(device)
        # Ensure 2-D adjacency (shared for the whole batch)
        if A.dim() == 3:      # [B,N,N] -> [N,N]
            A = A[0]
        mseq= batch["mask_seq"].to(device)     # [B,L,N]
        y   = batch["y_trf"].to(device)        # [B,N]
        ym  = batch["y_mask"].to(device)       # [B,N]

        with autocast_ctx:
            yhat = model(X, A, mseq)           # [B,N]
            loss = criterion(yhat, y, ym)      # masked loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                # clip in FP32
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches  += 1

    return total_loss / max(n_batches, 1)

def train_and_eval(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("logging", {}).get("seed", 42))
    _set_seed(seed)

    device = _device_from_cfg(cfg)
    out_dir = _out_dir_from_cfg(cfg)

    # Data
    bs = int(cfg["train"].get("batch_size", 16))
    train_loader, ds, idx = make_dataloader(cfg_path, split="train", batch_size=bs, shuffle=True)
    val_loader,   _,  _  = make_dataloader(cfg_path, split="val",   batch_size=bs, shuffle=False)
    test_loader,  _,  _  = make_dataloader(cfg_path, split="test",  batch_size=bs, shuffle=False)

    # Model
    n_features = ds.F_kept
    n_nodes = ds.N                          # <-- add
    model = make_model(
        cfg["train"]["model"],
        n_features=n_features,
        n_nodes=n_nodes,                    # <-- add
        **cfg.get("model", {})
    ).to(device)

    # Loss & Optim
    criterion = get_loss(cfg.get("train", {}).get("loss", {}))
    opt = optim.Adam(model.parameters(),
                     lr=float(cfg["train"].get("lr", 1e-3)),
                     weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    amp = bool(cfg["train"].get("amp", False))

    # Callbacks
    stopper = EarlyStopper(patience=int(cfg["train"].get("early_stop", {}).get("patience", 20)),
                           mode="min", min_delta=0.0)
    ckpt = CheckpointManager(out_dir)
    best_metric_name = cfg["train"].get("early_stop", {}).get("metric", "val_mae")
    surge_thr = cfg["train"].get("surge_threshold", None)
    if surge_thr is not None: surge_thr = float(surge_thr)

    # Save config snapshot
    with open(out_dir / "config.snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Train loop
    epochs = int(cfg["train"].get("epochs", 200))
    hist = []
    for ep in range(1, epochs + 1):
        train_loss = _epoch(train_loader, model, criterion, opt, device,
                            amp=amp, train=True, grad_clip=grad_clip)
        # Validation
        model.eval()
        v_losses = []
        mets_accum = []
        with torch.no_grad():
            for b in val_loader:
                X = b["X_seq"].to(device); A = b["A"].to(device); mseq = b["mask_seq"].to(device)
                y = b["y_trf"].to(device); ym = b["y_mask"].to(device)
                yhat = model(X, A, mseq)
                v_losses.append(float(criterion(yhat, y, ym).detach().cpu()))
                mets_accum.append(compute_metrics(yhat, y, ym, surge_threshold=surge_thr))
        val_loss = float(sum(v_losses)/max(len(v_losses),1)) if len(v_losses) else float("nan")
        # Aggregate metrics
        def _mean_dict(ds): 
            if not ds: return {}
            keys = ds[0].keys()
            return {k: float(sum(d[k] for d in ds)/len(ds)) for k in keys}
        val_metrics = _mean_dict(mets_accum) if mets_accum else {}

        # Choose early-stop metric (default val_mae)
        metric_name = cfg["train"].get("early_stop", {}).get("metric", "val_mae")
        # map friendly names -> what we have
        name_map = {
            "val_mae": "mae",
            "val_rmse": "rmse",
            "val_bias": "bias",
            "val_loss": None,
        }
        key = name_map.get(metric_name, "mae")
        sel = val_loss if key is None else val_metrics.get(key, float("inf"))
        is_best = stopper.step(sel)
        ckpt.save(model, opt, ep, sel, is_best=is_best)

        hist.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}":v for k,v in val_metrics.items()}})
        print(f"[ep {ep:03d}] train_loss={train_loss:.4f}  val_mae={val_metrics.get('mae', float('nan')):.4f}  sel={sel:.4f}")

        if stopper.stopped:
            print(f"Early stopping at epoch {ep}. Best={stopper.best:.4f}")
            break

    # Load best and evaluate on test
    state = torch.load(ckpt.best_path, map_location="cpu")
    model.load_state_dict(state["model"]); model.to(device); model.eval()

    test_mets = []
    yh_list, y_list, ym_list, yr_list = [], [], [], []
    with torch.no_grad():
        for b in test_loader:
            X = b["X_seq"].to(device); A = b["A"].to(device); mseq = b["mask_seq"].to(device)
            y = b["y_trf"].to(device); ym = b["y_mask"].to(device)
            yhat = model(X, A, mseq)
            test_mets.append(compute_metrics(yhat, y, ym, surge_threshold=surge_thr))
            yh_list.append(yhat.cpu().numpy()); y_list.append(y.cpu().numpy()); ym_list.append(ym.cpu().numpy()); yr_list.append(b["year"].cpu().numpy())
    def _mean(ds): 
        keys = ds[0].keys(); return {k: float(sum(d[k] for d in ds)/len(ds)) for k in keys}
    test_metrics = _mean(test_mets) if test_mets else {}

    # Save artifacts
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": hist, "test": test_metrics}, f, indent=2)
    import numpy as np
    np.savez(out_dir / "predictions_test.npz",
             yhat=np.concatenate(yh_list, axis=0) if yh_list else np.empty((0,)),
             y=np.concatenate(y_list, axis=0) if y_list else np.empty((0,)),
             y_mask=np.concatenate(ym_list, axis=0) if ym_list else np.empty((0,)),
             years=np.concatenate(yr_list, axis=0) if yr_list else np.empty((0,)),
             )

    print("\n[Step 5] Test metrics:", test_metrics)
    print(f"[Step 5] Artifacts saved to: {out_dir}")

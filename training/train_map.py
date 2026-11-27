"""3-4 보정 지도 모델 학습 (학습설계 2·3).

사용:
  py -3.12 -m training.train_map --name smoke --steps 300             # 짧게 돌려 보기
  py -3.12 -m training.train_map --name v1 --steps 40000              # 본 학습 (이어서: --resume)

손실 (사람 영역 안에서만, 배경은 규칙 단계가 맡음):
  지도   : 유동화 L1 · 피부 비율 L1 · 톤 배율 L1(1/4 해상도) · 메움 BCE
  결과   : 예측 지도를 원본에 적용(apply_maps, 배치는 정답 것)한 것 vs 수정본 L1. 메움 자리는 뺌
           → 정답 지도를 정확히 못 뽑은 곳도 결과로 배운다
결과: C:\\photostory_data\\runs\\{name}\\  last.pt, best.pt, log.csv, previews/
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import MapDataset, list_items
from .model import CorrectionMapNet, apply_maps
from .safety import DEFAULT_SOURCE, DEFAULT_WORK, check_output

LAMBDA = dict(flow=0.1, alpha=1.0, gain=4.0, fill=0.2, recon=5.0)


def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (x * m).sum() / m.sum().clamp_min(1.0)


def losses(pred: dict, b: dict) -> dict[str, torch.Tensor]:
    m = b["region"]
    out = {}
    out["flow"] = masked_mean((pred["flow"] - b["flow"]).abs().sum(1, keepdim=True), m)
    out["alpha"] = masked_mean((pred["alpha"] - b["alpha"]).abs(), m)
    g = F.avg_pool2d(pred["gain"], 2)
    out["gain"] = masked_mean((g - b["gain"]).abs().mean(1, keepdim=True), F.avg_pool2d(m, 2))
    pw = torch.tensor(5.0, device=m.device)
    bce = F.binary_cross_entropy_with_logits(pred["fill"], b["fill"], pos_weight=pw, reduction="none")
    out["fill"] = masked_mean(bce, m)
    rec = apply_maps(b["rgb"], b["s1"], pred["flow"], b["affine"], pred["gain"], pred["alpha"])
    out["recon"] = masked_mean((rec - b["ret"]).abs().mean(1, keepdim=True), m * (1 - b["fill"]))
    out["total"] = sum(LAMBDA[k] * v for k, v in out.items())
    return out


@torch.no_grad()
def evaluate(net, loader, device, max_batches: int = 0) -> dict[str, float]:
    """평균 손실 + 결과 오차(0~255): 원본 그대로 / 정답 지도 적용 / 예측 지도 적용."""
    net.eval()
    acc: dict[str, float] = {}
    n = 0
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = net(b["rgb"], b["aux"])
        pred = {k: v.float() for k, v in pred.items()}
        ls = losses(pred, b)
        m = b["region"] * (1 - b["fill"])
        err = lambda x: float(masked_mean((x - b["ret"]).abs().mean(1, keepdim=True), m)) * 255  # noqa: E731
        vals = {k: float(v) for k, v in ls.items()}
        vals["err_raw"] = err(b["rgb"])
        vals["err_target"] = err(apply_maps(b["rgb"], b["s1"], b["flow"], b["affine"], b["gain"], b["alpha"]))
        vals["err_pred"] = err(apply_maps(b["rgb"], b["s1"], pred["flow"], b["affine"], pred["gain"], pred["alpha"]))
        for k, v in vals.items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1
    net.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


@torch.no_grad()
def save_previews(net, ds: MapDataset, keys: list[str], device, path: Path) -> None:
    """원본 | 수정본 | 예측 적용 | 정답 피부 | 예측 피부 | 정답 유동화 크기 | 예측 유동화 크기 | 예측 메움."""
    net.eval()
    rows = []
    for k in keys:
        s = ds.load(k)
        b = {n: v[None].to(device) for n, v in s.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            p = net(b["rgb"], b["aux"])
        p = {n: v.float() for n, v in p.items()}
        rec = apply_maps(b["rgb"], b["s1"], p["flow"], b["affine"], p["gain"], p["alpha"])
        img = lambda t: (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # noqa: E731
        def cmap(t, vmax):
            g = (t[0, 0].clamp(0, vmax) / vmax * 255).byte().cpu().numpy()
            return cv2.cvtColor(cv2.applyColorMap(g, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        mag = lambda f: f.pow(2).sum(1, keepdim=True).sqrt()  # noqa: E731
        rows.append(np.hstack([img(b["rgb"]), img(b["ret"]), img(rec), cmap(b["alpha"], 1), cmap(p["alpha"], 1),
                               cmap(mag(b["flow"]), 8), cmap(mag(p["flow"]), 8), cmap(torch.sigmoid(p["fill"]), 1)]))
    sheet = cv2.resize(np.vstack(rows), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    net.train()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    work = check_output(a.work, (DEFAULT_SOURCE,))
    run = check_output(work / "runs" / a.name, (DEFAULT_SOURCE,))
    run.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_keys, val_keys = list_items(work, "train"), list_items(work, "val")
    if len(val_keys) < 20:          # 정답 지도가 아직 앞쪽(오래된) 짝만 있을 때: 학습 쪽 끝 5%를 잠시 검증으로
        cut = max(20, len(train_keys) // 20)
        train_keys, val_keys = train_keys[:-cut], train_keys[-cut:]
        print(f"검증 짝이 아직 적어서 학습 쪽 {cut}쌍을 검증으로 씀", flush=True)
    print(f"학습 {len(train_keys)} / 검증 {len(val_keys)}", flush=True)
    tr = MapDataset(work, train_keys, augment=True)
    va = MapDataset(work, val_keys)
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=a.workers > 0)
    vl = DataLoader(va, a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    net = CorrectionMapNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    warm = min(1000, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    step, best = 0, float("inf")
    if a.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck.get("best", best)
        print(f"이어서: {step}걸음부터", flush=True)

    log_path = run / "log.csv"
    log_new = not log_path.exists()
    logf = open(log_path, "a", newline="", encoding="utf-8")
    logw = None
    preview_keys = val_keys[:4]
    t0, run_acc, run_n = time.perf_counter(), {}, 0
    net.train()
    while step < a.steps:
        for b in tl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = net(b["rgb"], b["aux"])
            pred = {k: v.float() for k, v in pred.items()}
            ls = losses(pred, b)
            opt.zero_grad(set_to_none=True)
            ls["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in ls.items():
                run_acc[k] = run_acc.get(k, 0.0) + float(v)
            run_n += 1
            if step % 50 == 0:
                el = time.perf_counter() - t0
                msg = " ".join(f"{k} {v / run_n:.4f}" for k, v in run_acc.items())
                print(f"[{step}/{a.steps}] {msg}  lr {sched.get_last_lr()[0]:.1e}  {el / 50:.2f}초/걸음", flush=True)
                t0, run_acc, run_n = time.perf_counter(), {}, 0
            if step % a.val_every == 0 or step == a.steps:
                ev = evaluate(net, vl, device, a.val_batches)
                print(f"  검증 {step}: 결과 오차 원본 {ev['err_raw']:.2f} → 정답 지도 {ev['err_target']:.2f}"
                      f" / 예측 지도 {ev['err_pred']:.2f}  (손실 {ev['total']:.4f})", flush=True)
                row = {"step": step, **ev}
                if logw is None:
                    logw = csv.DictWriter(logf, fieldnames=list(row))
                    if log_new:
                        logw.writeheader()
                logw.writerow(row)
                logf.flush()
                ck = dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                          best=min(best, ev["err_pred"]), args=vars(a) | {"work": str(a.work)})
                torch.save(ck, run / "last.pt")
                if ev["err_pred"] < best:
                    best = ev["err_pred"]
                    torch.save(ck, run / "best.pt")
                save_previews(net, va, preview_keys, device, run / "previews" / f"{step:06d}.jpg")
            if step >= a.steps:
                break
    logf.close()
    print(f"끝. 가장 좋은 검증 결과 오차 {best:.2f} → {run / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

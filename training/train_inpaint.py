"""3-5 부분 생성형(구멍 채우기) 학습 (학습설계 2 · 3-5).

사용:
  py -3.12 -m training.train_inpaint --name smoke_inp --steps 200
  py -3.12 -m training.train_inpaint --name inp1 --steps 60000        # 본 학습 (이어서: --resume)

손실: 구멍 안 L1 ×6 + 구멍 밖 L1 ×1 + 구멍 안 기울기 L1 ×2 + (gan-start 걸음부터) PatchGAN 힌지 ×gan
검증: 같은 조각·구멍 고정. 구멍 안 평균 오차(0~255) — 모델(err_pred) vs OpenCV 고전 채우기(err_base)
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

from .inpaint_data import InpaintDataset, keys_with_fill, list_items, telea_baseline
from .inpaint_model import InpaintNet, PatchDisc, compose
from .safety import DEFAULT_SOURCE, DEFAULT_WORK, check_output


def grad(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def masked_mean(x, m):
    return (x * m).sum() / (m.sum() * x.shape[1] / m.shape[1]).clamp_min(1.0)


def gen_losses(pred, b, d_fake=None, gan_w=0.0) -> dict[str, torch.Tensor]:
    img, m, lw = b["img"], b["mask"], b["loss"]
    l1 = (pred - img).abs()
    out = dict(hole=masked_mean(l1, m * lw), valid=masked_mean(l1, (1 - m) * lw))
    gp, gi = grad(pred), grad(img)
    mx, my = m[..., :, 1:] * m[..., :, :-1], m[..., 1:, :] * m[..., :-1, :]
    out["grad"] = masked_mean((gp[0] - gi[0]).abs(), mx) + masked_mean((gp[1] - gi[1]).abs(), my)
    out["gan"] = -d_fake.mean() if d_fake is not None else torch.zeros((), device=img.device)
    out["total"] = 6 * out["hole"] + out["valid"] + 2 * out["grad"] + gan_w * out["gan"]
    return out


@torch.no_grad()
def evaluate(net, loader, device, base: float) -> dict[str, float]:
    net.eval()
    s, n = 0.0, 0.0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = net(b["img"], b["mask"])
        m = b["mask"] * b["loss"]
        s += float(((pred.float() - b["img"]).abs().mean(1, keepdim=True) * m).sum())
        n += float(m.sum())
    net.train()
    return dict(err_pred=255 * s / max(n, 1), err_base=base)


def baseline(ds) -> float:
    s, n = 0.0, 0.0
    for i in range(len(ds)):
        b = ds[i]
        img = b["img"].numpy().transpose(1, 2, 0)
        m = (b["mask"][0] * b["loss"][0]).numpy()
        out = telea_baseline(img, m)
        s += float((np.abs(out - img).mean(2) * m).sum())
        n += float(m.sum())
    return 255 * s / max(n, 1)


@torch.no_grad()
def save_previews(net, ds, idx, device, path: Path):
    net.eval()
    rows = []
    for i in idx:
        b = ds[i]
        img, m = b["img"][None].to(device), b["mask"][None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = net(img, m)
        c = compose(img, m, pred.float())
        u8 = lambda t: (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # noqa: E731
        holed = u8(img * (1 - m) + m * torch.tensor([1.0, 0, 1.0], device=device)[None, :, None, None])
        rows.append(np.hstack([holed, u8(c), u8(img)]))
    net.train()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--name", default="inp1")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gan", type=float, default=0.05)
    ap.add_argument("--gan-start", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-items", type=int, default=400)
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

    tk = keys_with_fill(work, list_items(work, "train"))
    vk = keys_with_fill(work, list_items(work, "val"))
    if len(vk) < 20:
        cut = max(20, len(tk) // 20)
        tk, vk = tk[:-cut], tk[-cut:]
    vk = vk[:a.val_items]
    print(f"학습 {len(tk)} / 검증 {len(vk)}", flush=True)
    tr = InpaintDataset(work, tk, a.patch, augment=True)
    va = InpaintDataset(work, vk, a.patch, fixed_seed=1)
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=a.workers > 0)
    vl = DataLoader(va, a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    net, disc = InpaintNet().to(device), PatchDisc().to(device)
    opt = torch.optim.AdamW(net.parameters(), a.lr, betas=(0.5, 0.99), weight_decay=1e-4)
    dopt = torch.optim.AdamW(disc.parameters(), a.lr, betas=(0.5, 0.99))
    warm = min(1000, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    step, best = 0, float("inf")
    base_path = run / "baseline.txt"
    if a.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device)
        net.load_state_dict(ck["model"])
        disc.load_state_dict(ck["disc"])
        opt.load_state_dict(ck["opt"])
        dopt.load_state_dict(ck["dopt"])
        sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck.get("best", best)
        print(f"이어서: {step}걸음부터", flush=True)
    if base_path.exists():
        base = float(base_path.read_text())
    else:
        base = baseline(va)
        base_path.write_text(f"{base:.4f}")
    print(f"고전 채우기(TELEA) 구멍 오차 {base:.2f}", flush=True)

    log_path = run / "log.csv"
    log_new = not log_path.exists()
    logf = open(log_path, "a", newline="", encoding="utf-8")
    logw = None
    t0, acc, n = time.perf_counter(), {}, 0
    net.train()
    while step < a.steps:
        for b in tl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            use_gan = step >= a.gan_start and a.gan > 0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = net(b["img"], b["mask"])
            pred = pred.float()
            fake = compose(b["img"], b["mask"], pred)
            if use_gan:                                   # 판별기 한 걸음 (힌지)
                dl = F.relu(1 - disc(b["img"], b["mask"])).mean() + F.relu(1 + disc(fake.detach(), b["mask"])).mean()
                dopt.zero_grad(set_to_none=True)
                dl.backward()
                dopt.step()
            ls = gen_losses(pred, b, disc(fake, b["mask"]) if use_gan else None, a.gan if use_gan else 0.0)
            opt.zero_grad(set_to_none=True)
            ls["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in ls.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach())
            n += 1
            if step % 50 == 0:
                el = time.perf_counter() - t0
                msg = " ".join(f"{k} {v / n:.4f}" for k, v in acc.items())
                print(f"[{step}/{a.steps}] {msg}  lr {sched.get_last_lr()[0]:.1e}  {el / 50:.2f}초/걸음", flush=True)
                t0, acc, n = time.perf_counter(), {}, 0
            if step % a.val_every == 0 or step == a.steps:
                ev = evaluate(net, vl, device, base)
                print(f"  검증 {step}: 구멍 오차 고전 채우기 {base:.2f} / 모델 {ev['err_pred']:.2f}", flush=True)
                row = {"step": step, **ev}
                if logw is None:
                    logw = csv.DictWriter(logf, fieldnames=list(row))
                    if log_new:
                        logw.writeheader()
                logw.writerow(row)
                logf.flush()
                ck = dict(model=net.state_dict(), disc=disc.state_dict(), opt=opt.state_dict(),
                          dopt=dopt.state_dict(), sched=sched.state_dict(), step=step,
                          best=min(best, ev["err_pred"]), args=vars(a) | {"work": str(a.work)})
                torch.save(ck, run / "last.pt")
                if ev["err_pred"] < best:
                    best = ev["err_pred"]
                    torch.save(ck, run / "best.pt")
                save_previews(net, va, list(range(min(6, len(va)))), device, run / "previews" / f"{step:06d}.jpg")
            if step >= a.steps:
                break
    logf.close()
    print(f"끝. 가장 좋은 검증 구멍 오차 {best:.2f} (고전 채우기 {base:.2f}) → {run / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

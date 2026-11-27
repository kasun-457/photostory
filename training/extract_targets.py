"""3-3 정답 지도 뽑기 (학습설계 2·3).

잘라 낸 원본(o)·수정본(r) 짝 하나에서, 성현님이 한 보정을 지도 네 개로 나눠 적는다.

  ⓪ 배치    affine: 몸 전체를 옮기거나 키운 것 (닮음 변환 하나). 규칙(규격 배치)이 맡으므로 학습하지 않고 적어만 둠
  ① 유동화  flow  : 배치를 뺀 나머지. 수정본 화소 x는 원본의 A·x + flow(x)에서 왔다 (RAFT, 부드럽게 다듬음)
  ② 톤 배율 gain  : 원본을 ①로 옮긴 o_w에 화소마다 곱할 색 배율 (저주파, RGB 3채널, log로 저장).
                    닷징·버닝, 머리카락 밝기·색이 여기 들어간다
  ③ 피부    alpha : 톤을 맞춘 t에 '수정-01' 스냅숏 1(S1)을 섞는 비율 (0~1) = 작업 내역 브러시. 피부 영역에만
                    r ≈ t + alpha · (S1(t) − t)
  ④ 메움    fill  : ①~③으로 설명되지 않고 남은 곳 (잔머리 정리, 콧볼 옆 그림자, 큰 잡티, 핀 등)
                    → 부분 생성형 모델이 채울 곳. 정답 화소는 수정본 그대로

배경(사람 영역 밖)은 규칙으로 이미 정확하므로 뽑지 않는다 (region 밖은 0).

결과: targets/{key}.npz  (flow: H/2×W/2×2 float16 조각 px, gain: H/4×W/4×3 float16 log,
      alpha: H/2×W/2 uint8, fill: H×W uint8, region: H/2×W/2 uint8)
      targets.csv  설명된 정도 (단계별 평균 오차)

사용:
  py -3.12 -m training.extract_targets --limit 20 --preview   # 시험 + 눈으로 볼 그림
  py -3.12 -m training.extract_targets                        # 전체 (이어서 하기 됨, 자르기와 동시에 돌려도 됨)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from .index_pairs import resume_manifest
from .safety import DEFAULT_SOURCE, DEFAULT_WORK, check_output

FLOW_DOWN = 2            # RAFT는 조각의 1/2 해상도에서 (메모리·속도, 유동화는 부드러움)
FLOW_SMOOTH = 2.0        # 1/2 해상도 px. 피부 결이 바뀐 것을 움직임으로 착각하지 않게
GAIN_SIGMA = 6.0         # 조각 px. 톤 배율의 부드러움
ALPHA_SIGMA = 3.0        # 조각 px. 피부 비율을 푸는 창
FILL_THRESH = 8.0        # 남은 오차(0~255)가 이보다 크면 메움 후보
FILL_MIN_AREA = 24       # 조각 px². 이보다 작은 점은 버림 (JPEG·흐름 잡음)
REGION_GROW = 24         # 사람 영역을 이만큼 넓혀서 (윤곽 유동화가 배경 쪽으로 밀린 것까지)
CROP_FACE_K = 384 / 1200  # 조각 안 얼굴 폭 384 → 가이드 px 배율


@dataclass
class TargetRow:
    key: str
    status: str = "ok"
    err_raw: float = 0.0       # |r − o|      (사람 영역 평균, 0~255)
    err_flow: float = 0.0      # |r − o_w|
    err_tone: float = 0.0      # |r − t|
    err_skin: float = 0.0      # |r − (t + α(S1−t))|
    flow_max: float = 0.0      # 조각 px (배치 뺀 유동화)
    flow_mean: float = 0.0
    alpha_mean: float = 0.0    # 얼굴 피부 안 평균
    fill_frac: float = 0.0     # 사람 영역 중 메움 비율
    g_scale: float = 1.0       # 배치(몸 전체) 닮음 변환: 배율·회전(도)·이동(조각 px)
    g_rot: float = 0.0
    g_dx: float = 0.0
    g_dy: float = 0.0
    seconds: float = 0.0


class FlowNet:
    """torchvision RAFT large. 정답 뽑기 전용 (판매 프로그램에 들어가지 않음)."""

    def __init__(self, device: str | None = None):
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(self.device)

    def __call__(self, a: np.ndarray, b: np.ndarray, iters: int = 12) -> np.ndarray:
        """a의 화소가 b의 어디에 있는지 (H×W×2, a 기준 px)."""
        t = self.torch
        def prep(x):
            x = t.from_numpy(x).permute(2, 0, 1)[None].float().to(self.device)
            return x / 127.5 - 1.0
        with t.inference_mode(), t.autocast(self.device, enabled=self.device == "cuda"):
            flows = self.net(prep(a), prep(b), num_flow_updates=iters)
        return flows[-1][0].permute(1, 2, 0).float().cpu().numpy()


def masked_blur(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """m(0~1) 안쪽만으로 흐림 (정규화 합성곱). 영역 밖 값이 새어 들어오지 않게."""
    mm = m if x.ndim == 2 else m[..., None]
    num = cv2.GaussianBlur(x * mm, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    den = den if x.ndim == 2 else den[..., None]
    return num / np.maximum(den, 1e-4)


def remap(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def snapshot1_clean(t: np.ndarray) -> np.ndarray:
    """skin.snapshot1과 같은 식, 잡음 없이 (조각 배율)."""
    from photostory.steps.skin import snapshot1
    return snapshot1(np.clip(t, 0, 255).astype(np.uint8), CROP_FACE_K, 0.0)


def global_similarity(fl: np.ndarray, mask: np.ndarray, step: int = 8) -> np.ndarray:
    """흐름(r→o)에 가장 잘 맞는 닮음 변환 2×3 (r 좌표 → o 좌표). 유동화된 곳은 RANSAC이 버린다."""
    ys, xs = np.mgrid[0:fl.shape[0]:step, 0:fl.shape[1]:step]
    sel = mask[ys, xs]
    if sel.sum() < 20:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    dst = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
    src = dst + fl[ys[sel], xs[sel]]
    A, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=1.5,
                                       maxIters=4000, confidence=0.999)
    return A if A is not None else np.array([[1, 0, 0], [0, 1, 0]], np.float64)


def extract(o: np.ndarray, r: np.ndarray, person: np.ndarray, face_skin: np.ndarray | None,
            valid: np.ndarray, flownet: FlowNet, body_skin: np.ndarray | None = None) -> tuple[dict, dict]:
    """o, r: uint8 RGB (H×W). person·face_skin·body_skin·valid: uint8 0~255. → (지도들, 통계)."""
    skin_masks = (face_skin, body_skin)
    h, w = o.shape[:2]
    of, rf = o.astype(np.float32), r.astype(np.float32)
    reg = cv2.dilate((person > 128).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * REGION_GROW + 1,) * 2))
    reg = (reg & (valid > 128)).astype(np.float32)
    regf = cv2.GaussianBlur(reg, (0, 0), 4)
    inside = reg > 0.5

    # ① 유동화: r의 화소가 o의 어디에 있나 (1/2 해상도)
    hs, ws = h // FLOW_DOWN, w // FLOW_DOWN
    rs = cv2.resize(r, (ws, hs), interpolation=cv2.INTER_AREA)
    os_ = cv2.resize(o, (ws, hs), interpolation=cv2.INTER_AREA)
    fl = flownet(rs, os_)
    reg_s = cv2.resize(reg, (ws, hs), interpolation=cv2.INTER_AREA)
    # 몸 전체를 옮기거나 키운 것(배치)은 유동화가 아니다 → 닮음 변환 하나로 먼저 빼서 따로 적는다
    A = global_similarity(fl, reg_s > 0.5)
    gx, gy = np.meshgrid(np.arange(ws, dtype=np.float32), np.arange(hs, dtype=np.float32))
    g = np.dstack([A[0, 0] * gx + A[0, 1] * gy + A[0, 2] - gx,
                   A[1, 0] * gx + A[1, 1] * gy + A[1, 2] - gy]).astype(np.float32)
    loc = cv2.GaussianBlur(fl - g, (0, 0), FLOW_SMOOTH)
    loc *= cv2.resize(regf, (ws, hs), interpolation=cv2.INTER_AREA)[..., None]   # 배경 쪽은 0으로
    flow = cv2.resize(g + loc, (w, h), interpolation=cv2.INTER_LINEAR) * FLOW_DOWN
    ow = remap(of, flow)
    # 영역 마스크도 수정본 위치로 옮김
    reg = (remap(reg, flow) > 0.5).astype(np.float32)
    regf = cv2.GaussianBlur(reg, (0, 0), 4)
    inside = reg > 0.5
    skin = np.zeros((h, w), np.float32)
    for m in skin_masks:
        if m is not None:
            skin = np.maximum(skin, m.astype(np.float32) / 255)
    skin = cv2.GaussianBlur(remap(skin, flow), (0, 0), 2) * regf

    # ② 톤 배율 (영역 안 저주파 비)
    lr = masked_blur(rf + 1.0, reg, GAIN_SIGMA)
    lo = masked_blur(ow + 1.0, reg, GAIN_SIGMA)
    gain = np.clip(np.where(lo > 0.5, lr / np.maximum(lo, 0.5), 1.0), 0.5, 2.0)   # 영역에서 먼 곳은 1
    gain = gain * regf[..., None] + (1 - regf[..., None])
    t = ow * gain

    # ③ 피부 비율: (r − t) ≈ α (S1 − t)  (창 안 최소제곱)
    s1 = snapshot1_clean(t)
    d, e = (rf - t).sum(2), (s1 - t).sum(2)
    num = cv2.GaussianBlur(d * e, (0, 0), ALPHA_SIGMA)
    den = cv2.GaussianBlur(e * e, (0, 0), ALPHA_SIGMA)
    alpha = np.clip(num / np.maximum(den, 1.0), 0, 1) * (den > 9.0) * skin   # 피부에만 (수정-01 브러시)
    rec = t + alpha[..., None] * (s1 - t)

    # ④ 메움: 아직 남은 오차
    err = np.abs(rf - rec).mean(2)
    cand = (cv2.GaussianBlur(err, (0, 0), 1.5) > FILL_THRESH) & inside
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= FILL_MIN_AREA
    fill = keep[lab]

    def mean_err(x):
        return float(np.abs(rf - x).mean(2)[inside].mean()) if inside.any() else 0.0

    lf = cv2.resize(loc, (w, h), interpolation=cv2.INTER_LINEAR) * FLOW_DOWN
    mag = np.hypot(lf[..., 0], lf[..., 1])                     # 유동화(배치 뺀 것)만
    fsk = remap(face_skin.astype(np.float32), flow) if face_skin is not None else None
    fs = fsk is not None and (fsk > 128).any()
    stats = dict(err_raw=mean_err(of), err_flow=mean_err(ow), err_tone=mean_err(t), err_skin=mean_err(rec),
                 flow_max=float(mag.max()), flow_mean=float(mag[inside].mean()) if inside.any() else 0.0,
                 alpha_mean=float(alpha[fsk > 128].mean()) if fs else 0.0,
                 g_scale=float(np.hypot(A[0, 0], A[1, 0])), g_rot=float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
                 g_dx=float(A[0, 2] * FLOW_DOWN), g_dy=float(A[1, 2] * FLOW_DOWN),
                 fill_frac=float(fill[inside].mean()) if inside.any() else 0.0)
    maps = dict(
        flow=(loc * FLOW_DOWN).astype(np.float16),            # 유동화만, 1/2 해상도, 조각 px (r 기준 → o 위치)
        affine=np.array([[A[0, 0], A[0, 1], A[0, 2] * FLOW_DOWN], [A[1, 0], A[1, 1], A[1, 2] * FLOW_DOWN]]),  # 배치
        gain=np.log(cv2.resize(gain, (w // 4, h // 4), interpolation=cv2.INTER_AREA)).astype(np.float16),
        alpha=np.clip(cv2.resize(alpha, (ws, hs), interpolation=cv2.INTER_AREA) * 255 + 0.5, 0, 255).astype(np.uint8),
        fill=(fill * 255).astype(np.uint8),
        region=(cv2.resize(reg, (ws, hs), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8),
    )
    return maps, stats


def load_crop(crops: Path, key: str):
    from .crop_pairs import load_npz
    o = cv2.cvtColor(cv2.imread(str(crops / f"{key}_o.jpg")), cv2.COLOR_BGR2RGB)
    r = cv2.cvtColor(cv2.imread(str(crops / f"{key}_r.jpg")), cv2.COLOR_BGR2RGB)
    meta = load_npz(crops / f"{key}.npz")
    return o, r, meta


def preview(o, r, maps, path: Path) -> None:
    """원본 | 수정본 | 원본+흐름 화살표 | 피부 α | 톤(밝게 빨강·어둡게 파랑) | 메움."""
    h, w = o.shape[:2]
    fl = cv2.resize(maps["flow"].astype(np.float32), (w, h))     # 유동화만 (배치는 빼고 그림)
    arrows = o.copy()
    for y in range(40, h, 48):
        for x in range(40, w, 48):
            dx, dy = fl[y, x]
            if dx * dx + dy * dy > 1.0:
                cv2.arrowedLine(arrows, (x, y), (int(x - dx * 4), int(y - dy * 4)), (255, 0, 0), 2, tipLength=0.3)
    a = cv2.resize(maps["alpha"], (w, h))
    a = cv2.cvtColor(cv2.applyColorMap(a, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    g = cv2.resize(maps["gain"].astype(np.float32).mean(2), (w, h))
    tone = np.full((h, w, 3), 128, np.float32)
    tone[..., 0] += np.clip(g, 0, None) * 600
    tone[..., 2] += np.clip(-g, 0, None) * 600
    fill = cv2.cvtColor(maps["fill"], cv2.COLOR_GRAY2RGB)
    tiles = [o, r, arrows, a, np.clip(tone, 0, 255).astype(np.uint8), fill]
    img = np.hstack(tiles)
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="previews/에 눈으로 볼 그림도 저장")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    work = check_output(a.work, (DEFAULT_SOURCE,))
    crops = work / "crops"
    out = check_output(work / "targets", (DEFAULT_SOURCE,))
    out.mkdir(exist_ok=True)
    pv = work / "previews"
    if a.preview:
        pv.mkdir(exist_ok=True)
    manifest = work / "targets.csv"

    done = resume_manifest(manifest)
    with open(work / "crops.csv", newline="", encoding="utf-8-sig") as fh:
        keys = [r["key"] for r in csv.DictReader(fh) if r["status"] == "ok" and r["key"] not in done]
    if a.limit:
        keys = keys[:a.limit]
    print(f"할 것 {len(keys)}쌍 (이미 {len(done)})", flush=True)

    net = FlowNet()
    new = not manifest.exists()
    fh = open(manifest, "a", newline="", encoding="utf-8-sig")
    wr = csv.DictWriter(fh, fieldnames=[f.name for f in fields(TargetRow)])
    if new:
        wr.writeheader()
    t_start = time.perf_counter()
    with ThreadPoolExecutor(3) as io:
        loads = [io.submit(load_crop, crops, k) for k in keys[:3]]
        for i, key in enumerate(keys):
            t0 = time.perf_counter()
            row = TargetRow(key)
            try:
                o, r, meta = loads[i].result()
            except Exception as e:
                row.status = f"읽기실패:{type(e).__name__}"
                o = None
            if i + 3 < len(keys):
                loads.append(io.submit(load_crop, crops, keys[i + 3]))
            if o is not None:
                try:
                    ms = meta["masks"]
                    maps, stats = extract(o, r, ms["person"], ms.get("face_skin"), meta["valid"], net,
                                          ms.get("body_skin"))
                    np.savez_compressed(out / f"{key}.npz", **maps)
                    for k, v in stats.items():
                        setattr(row, k, v)
                    if a.preview:
                        io.submit(preview, o, r, maps, pv / f"{key}.jpg")
                except Exception as e:
                    row.status = f"오류:{type(e).__name__}:{e}"[:120]
            row.seconds = time.perf_counter() - t0
            wr.writerow(asdict(row))
            n = i + 1
            if n % 50 == 0 or n == len(keys):
                fh.flush()
                el = time.perf_counter() - t_start
                print(f"  {n}/{len(keys)}  {el / n:.2f}초/쌍  남은 약 {(len(keys) - n) * el / n / 60:.0f}분", flush=True)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

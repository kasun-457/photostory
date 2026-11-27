"""3-2 얼굴 맞춰 자르기 (학습설계 2).

index.csv의 짝마다 원본에서 얼굴 점을 찾고, 머리~어깨를 **같은 배율**(얼굴 폭 FW px)로 잘라 낸다.
수정본도 똑같은 상자로 자른다 (원본과 수정본은 크기가 같고 화소 위치가 맞음).

결과 (작업 폴더 아래, 이름 대신 짧은 번호를 씀):
  crops/{key}_o.jpg    원본 조각 (W×H, JPEG 100 · 4:4:4)
  crops/{key}_r.jpg    수정본 조각
  crops/{key}.npz      얼굴 점·부위 다각형(조각 좌표), 영역 마스크(사람·머리카락·얼굴 피부·몸 피부·옷), 유효 영역, 자른 상자
  crops.csv            key, 원래 짝 정보, 상태, 차이 통계

사용:
  py -3.12 -m training.crop_pairs --limit 20          # 시험
  py -3.12 -m training.crop_pairs --workers 4         # 전체 (이어서 하기 됨)

- 원본 폴더는 읽기만 한다 (safety).
- 원본 해상도로 다 풀지 않고 JPEG 축소 풀기(draft)로 1/2만 푼 뒤 줄인다. 원본·수정본은 같은 방법으로 푼다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .index_pairs import Pair, read_index, resume_manifest
from .safety import DEFAULT_SOURCE, DEFAULT_WORK, check_output, open_source

FW = 384                 # 조각 안의 얼굴 폭 (regions.face_w 기준, 원본 1200px 안팎 → 약 1/3)
CROP_W, CROP_H = 1408, 1728   # 얼굴 폭의 3.67배 × 4.5배 (머리 위 볼륨 ~ 어깨)
TOP_MARGIN = 1.0         # 이마_위에서 위로 얼굴 폭의 1.0배
JPEG_Q = 100


def pair_key(p: Pair) -> str:
    return hashlib.sha1(f"{p.folder}/{p.original}".encode("utf-8")).hexdigest()[:12]


def load_rgb(path: Path, reduce: int = 1) -> np.ndarray:
    """JPEG를 읽기 전용으로 열어 sRGB uint8로. reduce = 1·2·4·8 (JPEG 축소 풀기)."""
    from photostory.core.io.imageio import _pil_to_rgb, _to_srgb
    with open_source(path) as fh, Image.open(fh) as im:
        if reduce > 1:
            im.draft("RGB", (im.width // reduce, im.height // reduce))
        im.load()
        im = ImageOps.exif_transpose(im)
        im = _to_srgb(im)
        return _pil_to_rgb(im)


def crop_box(points: dict, img_w: int, img_h: int) -> tuple[float, float, float]:
    """(배율 s, 상자 왼쪽 x0, 위 y0) 원래 사진 px. 조각 = (원래 - (x0, y0)) * s."""
    from photostory.core.measure.regions import face_w
    fw = face_w(points)
    s = FW / fw
    cx = (points["왼눈_바깥꼭짓점"][0] + points["오른눈_바깥꼭짓점"][0]) / 2
    x0 = cx - CROP_W / 2 / s
    y0 = points["이마_위"][1] - TOP_MARGIN * fw
    return s, x0, y0


def warp_crop(rgb: np.ndarray, s: float, x0: float, y0: float, border=cv2.BORDER_REPLICATE) -> np.ndarray:
    """rgb(이미 k배로 풀린 상태가 아니라 원래 좌표계라고 보고) → CROP_W×CROP_H. 줄일 때는 영역 평균."""
    m = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)
    if s < 1:
        # 먼저 정수 배율 근처까지 영역 평균으로 줄이고 나머지를 선형으로 (계단 현상 방지)
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        sx, sy = small.shape[1] / w, small.shape[0] / h
        m = np.array([[s / sx, 0, -x0 * s], [0, s / sy, -y0 * s]], np.float64)
        rgb = small
    return cv2.warpAffine(rgb, m, (CROP_W, CROP_H), flags=cv2.INTER_LINEAR, borderMode=border)


@dataclass
class CropRow:
    key: str
    folder: str
    original: str
    retouched: str
    date: str
    split: str
    glasses: bool
    status: str = "ok"         # ok / 얼굴없음 / 읽기실패 / 오류
    scale: float = 0.0         # 원래 px → 조각 px
    x0: float = 0.0
    y0: float = 0.0
    valid: float = 0.0         # 조각 중 사진 안쪽 비율
    diff_face: float = 0.0     # 얼굴 피부에서 원본·수정본 평균 절대 차 (0~255)
    diff_all: float = 0.0      # 사람 영역 전체
    seconds: float = 0.0


class Cropper:
    def __init__(self, src: Path, out: Path):
        from photostory.models.hub import ModelHub
        self.src, self.out = src, out
        self.hub = ModelHub()
        self.face = self.hub.get("face_landmarks")
        self.seg = self.hub.get("person_segmentation")

    def run(self, p: Pair) -> CropRow:
        t0 = time.perf_counter()
        key = pair_key(p)
        row = CropRow(key, p.folder, p.original, p.retouched, p.date, p.split, p.glasses)
        d = self.src / p.folder
        try:
            det = load_rgb(d / p.original, 4)                 # 얼굴 찾기는 1/4로 빠르게
        except OSError:
            row.status = "읽기실패"
            return row
        kd = p.width / det.shape[1] if p.width else 1.0
        f = self.face.detect(det)
        if f is None or "이마_위" not in f.points:
            row.status = "얼굴없음"
            return row
        s, _, _ = crop_box({n: (x * kd, y * kd) for n, (x, y) in f.points.items()}, 0, 0)
        reduce = 2 if s <= 0.5 else 1                         # 줄일 만큼만 덜 풀기 (키우지 않게)
        try:
            o = load_rgb(d / p.original, reduce)
            r = load_rgb(d / p.retouched, reduce)
        except OSError:
            row.status = "읽기실패"
            return row
        if o.shape != r.shape:
            row.status = "크기다름"
            return row
        k = p.width / o.shape[1] if p.width else 1.0      # 풀린 배율
        f2 = self.face.detect(o)                          # 점은 풀린 영상에서 다시 (더 정확)
        if f2 is not None and "이마_위" in f2.points:
            f = f2
        else:
            k = kd                                        # 못 찾으면 1/4 결과를 그대로
        pts = {n: (x * k, y * k) for n, (x, y) in f.points.items()}     # 원래 사진 px
        s, x0, y0 = crop_box(pts, p.width or o.shape[1], p.height or o.shape[0])
        sd = s * k                                       # 풀린 영상 px → 조각 px
        co = warp_crop(o, sd, x0 / k, y0 / k)
        cr = warp_crop(r, sd, x0 / k, y0 / k)
        ones = np.full(o.shape[:2], 255, np.uint8)
        valid = warp_crop(ones, sd, x0 / k, y0 / k, border=cv2.BORDER_CONSTANT)

        seg = self.seg.segment(co)
        to_u8 = lambda m: None if m is None else np.clip(m * 255 + 0.5, 0, 255).astype(np.uint8)  # noqa: E731
        masks = {n: to_u8(getattr(seg, n)) for n in ("person", "hair", "face_skin", "body_skin", "clothes")}
        cpts = {n: ((x - x0) * s, (y - y0) * s) for n, (x, y) in pts.items()}
        regions = {n: (np.asarray(v, np.float64) * k - (x0, y0)) * s for n, v in f.regions.items()}

        diff = np.abs(co.astype(np.int16) - cr.astype(np.int16)).mean(axis=2)
        fs = masks["face_skin"]
        pm = masks["person"]
        row.diff_face = float(diff[fs > 128].mean()) if fs is not None and (fs > 128).any() else -1.0
        row.diff_all = float(diff[pm > 128].mean()) if (pm > 128).any() else -1.0
        row.scale, row.x0, row.y0 = s, x0, y0
        row.valid = float((valid > 128).mean())

        Image.fromarray(co).save(self.out / f"{key}_o.jpg", quality=JPEG_Q, subsampling=0)
        Image.fromarray(cr).save(self.out / f"{key}_r.jpg", quality=JPEG_Q, subsampling=0)
        np.savez_compressed(
            self.out / f"{key}.npz",
            point_names=np.array(list(cpts)), points=np.array(list(cpts.values()), np.float32),
            region_names=np.array(list(regions)),
            **{f"region_{i}": v.astype(np.float32) for i, v in enumerate(regions.values())},
            valid=valid, box=np.array([s, x0, y0], np.float64),
            **{f"mask_{n}": m for n, m in masks.items() if m is not None})
        row.seconds = time.perf_counter() - t0
        return row


def load_npz(path: Path) -> dict:
    """crops/{key}.npz → {'points': {이름: (x,y)}, 'regions': {이름: (K,2)}, 'masks': {...}, 'valid', 'box'}."""
    z = np.load(path)
    pts = {str(n): tuple(map(float, xy)) for n, xy in zip(z["point_names"], z["points"])}
    regs = {str(n): z[f"region_{i}"] for i, n in enumerate(z["region_names"])}
    masks = {k[5:]: z[k] for k in z.files if k.startswith("mask_")}
    return {"points": pts, "regions": regs, "masks": masks, "valid": z["valid"], "box": z["box"]}


# ---- 여러 프로세스로 돌리기 -------------------------------------------------
_worker: Cropper | None = None


def _init(src: str, out: str):
    global _worker
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    cv2.setNumThreads(2)
    _worker = Cropper(Path(src), Path(out))


def _job(p: Pair) -> CropRow:
    try:
        return _worker.run(p)
    except Exception as e:  # 한 장 실패로 전체가 멈추지 않게
        r = CropRow(pair_key(p), p.folder, p.original, p.retouched, p.date, p.split, p.glasses)
        r.status = f"오류:{type(e).__name__}"
        return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    work = check_output(a.work, (a.src,))
    out = check_output(work / "crops", (a.src,))
    out.mkdir(parents=True, exist_ok=True)
    manifest = work / "crops.csv"

    done = resume_manifest(manifest)
    todo = [p for p in read_index(work / "index.csv") if p.status == "ok" and pair_key(p) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"할 것 {len(todo)}쌍 (이미 {len(done)})", flush=True)

    new = not manifest.exists()
    fh = open(manifest, "a", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(fh, fieldnames=[f.name for f in fields(CropRow)])
    if new:
        w.writeheader()
    t0 = time.perf_counter()
    from multiprocessing import Pool
    with Pool(a.workers, initializer=_init, initargs=(str(a.src), str(out))) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, todo, chunksize=4), 1):
            w.writerow(asdict(row))
            if i % 50 == 0 or i == len(todo):
                fh.flush()
                el = time.perf_counter() - t0
                print(f"  {i}/{len(todo)}  {el / i:.2f}초/쌍  남은 약 {(len(todo) - i) * el / i / 60:.0f}분", flush=True)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

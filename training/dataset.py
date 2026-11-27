"""3-4 학습 데이터: 잘라 낸 조각 + 정답 지도 → 텐서 (학습설계 3).

모든 것은 조각의 1/2 해상도(704×864)에서 다룬다. 톤 배율만 1/4 해상도(352×432).

한 개 =
  rgb     (3,H,W)   원본 0~1
  aux     (12,H,W)  얼굴 부위 6 (윤곽·눈·눈썹·입술·코·턱선) + 영역 5 (사람·머리카락·얼굴 피부·몸 피부·옷) + 유효
  ret     (3,H,W)   수정본 0~1 (결과 손실용)
  s1      (3,H,W)   원본의 '수정-01' 스냅숏 1 (결과 손실용)
  flow    (2,H,W)   유동화 (1/2 해상도 px, 배치 뺀 것)
  affine  (2,3)     배치 (1/2 해상도 좌표)
  alpha   (1,H,W)   피부 비율 0~1
  gain    (3,H/2,W/2) 톤 배율 log
  fill    (1,H,W)   메움 0/1
  region  (1,H,W)   손실을 볼 곳 0~1
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .crop_pairs import CROP_H, CROP_W, load_npz

H, W = CROP_H // 2, CROP_W // 2
PART_GROUPS = (("얼굴윤곽",), ("왼눈", "오른눈"), ("왼눈썹", "오른눈썹"), ("입술",),
               ("코아래", "콧대"), ("왼턱선_선", "오른턱선_선"))
LINES = {"콧대", "왼턱선_선", "오른턱선_선"}
MASKS = ("person", "hair", "face_skin", "body_skin", "clothes")
N_AUX = len(PART_GROUPS) + len(MASKS) + 1


def list_items(work: Path, split: str | None = None) -> list[str]:
    """정답 지도까지 있는 key 목록. split = 'train' / 'val' / None."""
    with open(work / "crops.csv", newline="", encoding="utf-8-sig") as fh:
        sp = {r["key"]: r["split"] for r in csv.DictReader(fh) if r["status"] == "ok"}
    with open(work / "targets.csv", newline="", encoding="utf-8-sig") as fh:
        ok = [r["key"] for r in csv.DictReader(fh) if r["status"] == "ok"]
    return [k for k in ok if k in sp and (split is None or sp[k] == split)]


def render_parts(regions: dict, scale: float = 0.5) -> np.ndarray:
    out = np.zeros((len(PART_GROUPS), H, W), np.float32)
    for i, group in enumerate(PART_GROUPS):
        m = np.zeros((H, W), np.uint8)
        for name in group:
            if name not in regions:
                continue
            p = np.round(np.asarray(regions[name]) * scale).astype(np.int32).reshape(-1, 1, 2)
            if name in LINES:
                cv2.polylines(m, [p], False, 255, 3, cv2.LINE_AA)
            else:
                cv2.fillPoly(m, [p], 255, cv2.LINE_AA)
        out[i] = m / 255.0
    return out


def snapshot1_half(rgb_u8: np.ndarray) -> np.ndarray:
    from photostory.steps.skin import snapshot1
    return snapshot1(rgb_u8, 384 / 1200 / 2, 0.0)


def flip_affine(A: np.ndarray, w: int) -> np.ndarray:
    """좌우 뒤집은 좌표계에서의 같은 변환. F·A·F, F: x → (w−1) − x."""
    F = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], np.float64)
    A3 = np.vstack([A, [0, 0, 1]])
    return (F @ A3 @ F)[:2]


class MapDataset(Dataset):
    def __init__(self, work: Path, keys: list[str], augment: bool = False):
        self.work, self.keys, self.augment = Path(work), keys, augment

    def __len__(self):
        return len(self.keys)

    def load(self, key: str, flip: bool = False) -> dict:
        crops, tg = self.work / "crops", self.work / "targets"
        o = cv2.cvtColor(cv2.imread(str(crops / f"{key}_o.jpg")), cv2.COLOR_BGR2RGB)
        r = cv2.cvtColor(cv2.imread(str(crops / f"{key}_r.jpg")), cv2.COLOR_BGR2RGB)
        o = cv2.resize(o, (W, H), interpolation=cv2.INTER_AREA)
        r = cv2.resize(r, (W, H), interpolation=cv2.INTER_AREA)
        meta = load_npz(crops / f"{key}.npz")
        t = np.load(tg / f"{key}.npz")

        aux = np.zeros((N_AUX, H, W), np.float32)
        aux[:len(PART_GROUPS)] = render_parts(meta["regions"])
        for i, n in enumerate(MASKS):
            m = meta["masks"].get(n)
            if m is not None:
                aux[len(PART_GROUPS) + i] = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA) / 255.0
        aux[-1] = cv2.resize(meta["valid"], (W, H), interpolation=cv2.INTER_AREA) / 255.0

        s1 = snapshot1_half(o)
        flow = t["flow"].astype(np.float32) / 2                     # 조각 px → 1/2 해상도 px
        A = t["affine"].astype(np.float64).copy()
        A[:, 2] /= 2
        fill = t["fill"].reshape(H, 2, W, 2).max(axis=(1, 3)).astype(np.float32) / 255
        d = dict(rgb=o.astype(np.float32) / 255, ret=r.astype(np.float32) / 255,
                 s1=np.clip(s1, 0, 255).astype(np.float32) / 255,
                 flow=flow, alpha=t["alpha"].astype(np.float32) / 255,
                 gain=t["gain"].astype(np.float32), fill=fill,
                 region=t["region"].astype(np.float32) / 255)
        if flip:
            for k in d:
                d[k] = d[k][:, ::-1]
            d["flow"] = d["flow"] * np.array([-1, 1], np.float32)
            aux = aux[:, :, ::-1]
            A = flip_affine(A, W)
        out = {k: torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1) if v.ndim == 3 else v[None]))
               for k, v in d.items()}                                 # 모두 H×W×C 또는 H×W
        out["aux"] = torch.from_numpy(np.ascontiguousarray(aux))
        out["affine"] = torch.from_numpy(A.astype(np.float32))
        return out

    def __getitem__(self, i):
        flip = self.augment and np.random.random() < 0.5
        return self.load(self.keys[i], flip)

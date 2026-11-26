"""13. 배경 (가이드 §12).

- 흰색: 배경 픽셀의 채널별 하위 1% 값을 흰색 입력값으로 하는 레벨을 '배경 층'에 적용하고,
  사람 영역은 원래대로 위에 얹는다. 적용 후 배경이 전부 255가 아니면 입력값을 더 낮춘다.
- 색 배경: 배경을 그 색으로 채운다. 경계의 원래 배경색 섞임을 걷어 내서 흰 테두리를 줄인다.

수정본은 한 장이므로, 고른 규격의 배경색이 모두 같으면 그 색, 다르면 흰색으로 만든다.
다른 색 규격의 전송본은 transfer 단계에서 같은 사람 영역으로 배경만 바꿔 넣는다.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.pipeline import Frame, Step

BG_THR = 0.05        # 이보다 작으면 배경으로 본다
HARD_THR = 0.02      # 이보다 작으면 배경색으로 확정


def background_rgb_for(specs) -> tuple[int, int, int]:
    colors = {s.bg_rgb for s in specs}
    return colors.pop() if len(colors) == 1 else (255, 255, 255)


def _percentile_hist(hist: np.ndarray, pct: float) -> float:
    """np.percentile(값들, pct)와 같은 값 (선형 보간)을 256칸 히스토그램으로 계산한다."""
    n = int(hist.sum())
    k = pct / 100.0 * (n - 1)
    i = int(np.floor(k))
    cum = np.cumsum(hist)
    lo = int(np.searchsorted(cum, i + 1))            # 정렬했을 때 i번째 값
    hi = int(np.searchsorted(cum, min(i + 2, n)))    # i+1번째 값
    return lo + (k - i) * (hi - lo)


def levels_white(img: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """배경 픽셀이 255가 되도록 채널별 흰색 입력값을 정해 레벨을 적용한다.

    원본 해상도(약 3천만 화소)에서도 빠르도록 히스토그램으로 백분위를 구하고 cv2.LUT로 적용한다.
    """
    bgm = bg.astype(np.uint8)
    n_bg = cv2.countNonZero(bgm)
    if n_bg == 0:
        return img, [255, 255, 255]
    hists = [cv2.calcHist([img], [c], bgm, [256], [0, 256]).ravel() for c in range(3)]
    out = img
    white = [255, 255, 255]
    for pct in (1.0, 0.3, 0.1, 0.02):
        white = [max(1, int(_percentile_hist(hists[c], pct))) for c in range(3)]
        lut = np.stack([np.clip(np.arange(256) * 255.0 / w, 0, 255).astype(np.uint8) for w in white], axis=-1)
        out = cv2.LUT(img, lut.reshape(1, 256, 3))
        # 레벨 뒤 255가 아닌 배경 화소 = 한 채널이라도 흰색 입력값보다 작은 화소
        full = cv2.inRange(img, tuple(white), (255, 255, 255))
        short = cv2.countNonZero(cv2.bitwise_and(bgm, cv2.bitwise_not(full) // 255))
        if short / n_bg < 0.001:
            break
    return out, white


def composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return cv2.blendLinear(fg, bg, a, 1.0 - a)


def fill_where(img: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    """mask(bool)인 화소를 color로 (제자리)."""
    fill = np.empty_like(img)
    fill[:] = color
    cv2.copyTo(fill, mask.astype(np.uint8), img)
    return img


def _edge_box(alpha: np.ndarray, lo: float = 0.05, hi: float = 0.95):
    edge = ((alpha > lo) & (alpha < hi)).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(edge)
    return (None if w == 0 else (slice(y, y + h), slice(x, x + w)))


def decontaminate(img: np.ndarray, alpha: np.ndarray, old_bg: np.ndarray) -> np.ndarray:
    """경계 픽셀에서 원래 배경색 섞임을 뺀 전경색 F = (I - (1-a)B) / a. 경계를 둘러싼 범위만 계산."""
    box = _edge_box(alpha)
    if box is None:
        return img
    al = alpha[box]
    a = np.clip(al, 0.0, 1.0)[..., None].astype(np.float32)
    edge = (al > 0.05) & (al < 0.95)
    f = img[box].astype(np.float32)
    est = (f - (1.0 - a) * old_bg.astype(np.float32)) / np.maximum(a, 0.05)
    out = img.copy()
    sub = out[box]
    sub[edge] = np.clip(est[edge], 0, 255).astype(np.uint8)
    return out


def replace_background(img: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int],
                       old_bg: tuple[int, int, int] | None = None) -> np.ndarray:
    fill = np.empty_like(img)
    fill[:] = color
    fg = decontaminate(img, alpha, np.array(old_bg, np.float32)) if old_bg else img
    out = composite(fg, fill, alpha)
    return fill_where(out, alpha < HARD_THR, color)


def _median_bg(img: np.ndarray, bg: np.ndarray) -> tuple[int, int, int]:
    """배경 화소의 채널별 중간값 (히스토그램으로)."""
    bgm = bg.astype(np.uint8)
    if cv2.countNonZero(bgm) == 0:
        return (255, 255, 255)
    return tuple(int(_percentile_hist(cv2.calcHist([img], [c], bgm, [256], [0, 256]).ravel(), 50.0))
                 for c in range(3))


class BackgroundStep(Step):
    id = "background"
    title = "배경"
    param_keys = ("bg_color",)

    def run(self, frame: Frame, analysis, params) -> Frame:
        m = frame.person_mask
        if m is None:
            frame.notes.append("배경: 사람 영역이 없어 건너뜀")
            return frame
        color = tuple(params.get("bg_color") or (255, 255, 255))
        img = frame.image
        bg = m < BG_THR
        if color == (255, 255, 255):
            leveled, white = levels_white(img, bg)
            out = fill_where(composite(img, leveled, m), m < HARD_THR, (255, 255, 255))
            frame.outputs["bg_levels"] = white
            frame.notes.append(f"배경: 흰색 레벨 입력값 R{white[0]} G{white[1]} B{white[2]}")
        else:
            old = _median_bg(img, bg)
            out = replace_background(img, m, color, old)
            frame.notes.append("배경: 규격 색으로 채움")
        frame.outputs["bg_color"] = color
        frame.image = out
        return frame

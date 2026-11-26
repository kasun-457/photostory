"""13. 배경 (가이드 §12).

- 흰색: 배경 픽셀의 채널별 하위 1% 값을 흰색 입력값으로 하는 레벨을 '배경 층'에 적용하고,
  사람 영역은 원래대로 위에 얹는다. 적용 후 배경이 전부 255가 아니면 입력값을 더 낮춘다.
- 색 배경: 배경을 그 색으로 채운다. 경계의 원래 배경색 섞임을 걷어 내서 흰 테두리를 줄인다.

수정본은 한 장이므로, 고른 규격의 배경색이 모두 같으면 그 색, 다르면 흰색으로 만든다.
다른 색 규격의 전송본은 transfer 단계에서 같은 사람 영역으로 배경만 바꿔 넣는다.
"""
from __future__ import annotations

import numpy as np

from ..core.pipeline import Frame, Step

BG_THR = 0.05        # 이보다 작으면 배경으로 본다
HARD_THR = 0.02      # 이보다 작으면 배경색으로 확정


def background_rgb_for(specs) -> tuple[int, int, int]:
    colors = {s.bg_rgb for s in specs}
    return colors.pop() if len(colors) == 1 else (255, 255, 255)


def levels_white(img: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """배경 픽셀이 255가 되도록 채널별 흰색 입력값을 정해 레벨을 적용한다."""
    pixels = img[bg]
    if pixels.size == 0:
        return img, [255, 255, 255]
    out = img
    for pct in (1.0, 0.3, 0.1, 0.02):
        white = [max(1, int(np.percentile(pixels[:, c], pct))) for c in range(3)]
        lut = [np.clip(np.arange(256) * 255.0 / w, 0, 255).astype(np.uint8) for w in white]
        out = np.stack([lut[c][img[..., c]] for c in range(3)], axis=-1)
        if (out[bg] < 255).any(axis=1).mean() < 0.001:
            break
    return out, white


def composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    out = fg.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def decontaminate(img: np.ndarray, alpha: np.ndarray, old_bg: np.ndarray) -> np.ndarray:
    """경계 픽셀에서 원래 배경색 섞임을 뺀 전경색 F = (I - (1-a)B) / a."""
    a = np.clip(alpha, 0.0, 1.0)[..., None].astype(np.float32)
    edge = (alpha > 0.05) & (alpha < 0.95)
    if not edge.any():
        return img
    f = img.astype(np.float32)
    est = (f - (1.0 - a) * old_bg.astype(np.float32)) / np.maximum(a, 0.05)
    out = f.copy()
    out[edge] = np.clip(est[edge], 0, 255)
    return out.astype(np.uint8)


def replace_background(img: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int],
                       old_bg: tuple[int, int, int] | None = None) -> np.ndarray:
    fill = np.empty_like(img)
    fill[:] = color
    fg = decontaminate(img, alpha, np.array(old_bg, np.float32)) if old_bg else img
    out = composite(fg, fill, alpha)
    out[alpha < HARD_THR] = color
    return out


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
            out = composite(img, leveled, m)
            out[m < HARD_THR] = 255
            frame.outputs["bg_levels"] = white
            frame.notes.append(f"배경: 흰색 레벨 입력값 R{white[0]} G{white[1]} B{white[2]}")
        else:
            old = tuple(int(v) for v in np.median(img[bg], axis=0)) if bg.any() else (255, 255, 255)
            out = replace_background(img, m, color, old)
            frame.notes.append("배경: 규격 색으로 채움")
        frame.outputs["bg_color"] = color
        frame.image = out
        return frame

"""17. A4 출력본 (가이드 §16, 설계서 4절).

- 2480×3508px, 300dpi, 흰색. 규격마다 한 줄. 첫 줄 y=22, 다음 줄 = 이전 y + 이전 높이 + 35
- 사진 x = 146 + i × (사진폭 + 30). 전송 dpi가 300이 아니면 실제 크기(mm) 유지한 채 300dpi로
- 사진마다 1px 바깥쪽 획 RGB(180,180,180)
- 라벨 `{이름} ({규격})` 58px(14pt), RGB(180,180,180), 시계 방향 90°, 첫 사진 왼쪽 여백(x 약 44~97)
- 마무리 밝기 +5
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core.pipeline import Frame, Step
from ..core.specs import A4_FIRST_X, A4_GAP, A4_PX, Spec

ROW_TOP = 22
ROW_GAP = 35
STROKE = (180, 180, 180)
LABEL_PX = 58                      # 14pt @300dpi
LABEL_CENTER_X = 70                # 가이드: 가로 약 44~97px
BRIGHTNESS = 5
DEFAULT_FONT = Path(__file__).resolve().parent.parent / "app" / "resources" / "fonts" / "NanumGothic-Bold.ttf"


def brightness_lut(amount: float) -> np.ndarray:
    """밝기 +amount. 검정(0)과 흰색(255)은 그대로 두고 중간 톤을 올리는 곡선.

    Photoshop '밝기/대비'(레거시 아님)의 정확한 식은 공개돼 있지 않아 **확실하지 않음**.
    실제 Photoshop 결과와 비교해서 맞춰야 한다 (설계서 8-4).
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    y = x + (amount / 255.0) * 4.0 * x * (1.0 - x)
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def to_300dpi(img: np.ndarray, spec: Spec) -> np.ndarray:
    if spec.dpi == 300:
        return img
    w, h = spec.size_px_300
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if spec.dpi > 300 else cv2.INTER_CUBIC)


@lru_cache(maxsize=8)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def label_image(text: str, font_path: str | Path, max_len: int | None = None) -> Image.Image:
    """가로로 쓴 라벨을 시계 방향 90° 돌린 회색조 마스크.

    글꼴마다 글자 폭이 달라서(가이드의 KoreanPGS-B보다 넓은 글꼴 등) 사진 높이를 넘으면
    넘지 않을 때까지 글자 크기를 줄인다.
    """
    size = LABEL_PX
    f = _font(str(font_path), size)
    l, t, r, b = f.getbbox(text)
    while max_len and r - l + 2 > max_len and size > 30:
        size -= 1
        f = _font(str(font_path), size)
        l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - l + 2, LABEL_PX + 4), 0)
    ImageDraw.Draw(im).text((-l + 1, (LABEL_PX + 4 - (b - t)) // 2 - t), text, font=f, fill=255)
    return im.rotate(-90, expand=True)


def build_a4(name: str, specs: list[Spec], transfers: dict[str, np.ndarray],
             font_path: str | Path | None) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    canvas = Image.new("RGB", A4_PX, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = ROW_TOP
    for spec in specs:
        img = transfers.get(spec.id)
        if img is None:
            continue
        img = to_300dpi(img, spec)
        h, w = img.shape[:2]
        count = spec.a4_count_resolved()
        if y + h > A4_PX[1]:
            notes.append(f"A4: {spec.label} 줄이 A4 아래로 넘쳐서 넣지 못함")
            break
        tile = Image.fromarray(img)
        for i in range(count):
            x = A4_FIRST_X + i * (w + A4_GAP)
            if x + w > A4_PX[0]:
                notes.append(f"A4: {spec.label} {count}장 중 {i}장만 들어감")
                break
            canvas.paste(tile, (x, y))
            draw.rectangle([x - 1, y - 1, x + w, y + h], outline=STROKE, width=1)
        if font_path and Path(font_path).exists():
            lab = label_image(f"{name} ({spec.label})", font_path, max_len=h)
            lx = LABEL_CENTER_X - lab.width // 2
            canvas.paste(Image.new("RGB", lab.size, STROKE), (lx, y - 1), lab)
        else:
            notes.append("A4: 라벨 글꼴이 없어 라벨을 넣지 못함")
        y += h + ROW_GAP
    arr = np.asarray(canvas)
    arr = brightness_lut(BRIGHTNESS)[arr]
    return arr, notes


class A4Step(Step):
    id = "a4"
    title = "A4 출력본"
    optional = False
    param_keys = ("name", "specs", "font")

    def run(self, frame: Frame, analysis, params) -> Frame:
        font = params.get("font") or DEFAULT_FONT
        a4, notes = build_a4(params.get("name", ""), params.get("specs") or [],
                             frame.outputs.get("transfers", {}), font)
        frame.outputs["a4"] = a4
        frame.notes.extend(notes)
        return frame

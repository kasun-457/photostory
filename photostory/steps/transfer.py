"""16. 전송본 (가이드 §15). 트리밍만 한다: 축소·배치·자르기 외의 보정은 하지 않는다.

- 정수리 → 정수리선, 얼굴 중심(두 눈 안쪽 끝 중간) → 세로 중심선
- 턱은 최소~최대선 사이. 기본은 최소선, chin_bias(0~1)로 최대선 쪽으로 (머리 볼륨이 클 때)
- 설계서 2-4 미세 조정: 얼굴 크기 ±10%, 위아래 ±5%
- 수정본 배경색과 규격 배경색이 다르면 같은 사람 영역으로 배경만 바꾼다 (background.py 설명)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.pipeline import Frame, Step
from ..core.session import Placement
from ..core.specs import Spec
from .background import replace_background


@dataclass
class TransferCheck:
    crown: float          # 전송본에서 정수리 y
    chin: float           # 전송본에서 턱 y
    center: float         # 전송본에서 중심 x
    head_cm: float        # 정수리~턱 (cm)
    ok: bool              # 규격 기준선 안인지


def place(frame: Frame, spec: Spec, pl: Placement) -> tuple[np.ndarray, np.ndarray | None, TransferCheck]:
    g = frame.outputs["guides"]
    head = max(1.0, g["chin_y"] - g["crown_y"])
    W, H = spec.size_px
    chin_target = spec.chin_min_line + pl.chin_bias * (spec.chin_max_line - spec.chin_min_line)
    s = (chin_target - spec.crown_line) / head * (1.0 + pl.scale)

    src = frame.image
    sh, sw = src.shape[:2]
    nw, nh = max(1, round(sw * s)), max(1, round(sh * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    small = cv2.resize(src, (nw, nh), interpolation=interp)
    sx, sy = nw / sw, nh / sh                     # 반올림 뒤 실제 배율
    mask = None
    if frame.person_mask is not None:
        mask = cv2.resize(frame.person_mask, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

    tx = spec.center_x - g["center_x"] * sx
    ty = spec.crown_line + pl.shift * H - g["crown_y"] * sy
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    bg = frame.outputs.get("bg_color", (255, 255, 255))
    out = cv2.warpAffine(small, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(int(v) for v in bg))
    m_out = None
    if mask is not None:
        m_out = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if tuple(bg) != spec.bg_rgb and m_out is not None:
        out = replace_background(out, m_out, spec.bg_rgb, tuple(bg))

    crown = g["crown_y"] * sy + ty
    chin = g["chin_y"] * sy + ty
    center = g["center_x"] * sx + tx
    tol = 0.05 / 2.54 * spec.dpi     # ±0.05cm (설계서 1절 결과 화면 표시)
    ok = (abs(crown - spec.crown_line) <= tol
          and spec.chin_min_line - tol <= chin <= spec.chin_max_line + tol
          and abs(center - spec.center_x) <= tol)
    check = TransferCheck(crown, chin, center, (chin - crown) / spec.dpi * 2.54, ok)
    return out, m_out, check


class TransferStep(Step):
    id = "transfer"
    title = "전송본"
    optional = False
    param_keys = ("specs", "placements")

    def run(self, frame: Frame, analysis, params) -> Frame:
        frame.outputs["retouched"] = frame.image
        transfers, checks = {}, {}
        placements = params.get("placements") or {}
        for spec in params.get("specs") or []:
            pl = placements.get(spec.id) or Placement()
            img, _, chk = place(frame, spec, pl)
            transfers[spec.id], checks[spec.id] = img, chk
        frame.outputs["transfers"] = transfers
        frame.outputs["transfer_checks"] = checks
        return frame

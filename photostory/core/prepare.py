"""prepare: 사진마다 1회 하는 무거운 분석 (설계서 3-1의 P1~P3).

사진이 업로드 칸에 들어가는 순간 시작할 수 있도록, 화면과 상관없이 호출할 수 있게 만든다.
P4(보정 지도)·P5(안경 짝 정렬)는 1·2단계에서 추가한다.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models.hub import ModelHub
from ..models.roles import REQUIRED_POINTS, FaceLandmarks, PersonSegmentation
from .io.imageio import read_image
from .measure import basic, catchlight, eyes, nose_mouth, sharpness, spots
from .pipeline import Analysis, CancelToken

log = logging.getLogger(__name__)


class FaceNotFound(Exception):
    pass


@dataclass
class QuickCheck:
    ok: bool
    message: str = ""


def quick_face_check(rgb: np.ndarray, hub: ModelHub) -> QuickCheck:
    """업로드 칸 미리보기용: 얼굴이 있는지만 빠르게 본다 (설계서 1-2)."""
    face = hub.get("face_landmarks", FaceLandmarks).detect(rgb)
    if face is None:
        return QuickCheck(False, "얼굴을 찾지 못했습니다")
    if face.extra.get("faces", 1) > 1:
        return QuickCheck(True, "얼굴이 여러 개입니다. 가장 큰 얼굴을 씁니다")
    return QuickCheck(True)


def analyze(path: Path | str, hub: ModelHub, token: CancelToken | None = None,
            rgb: np.ndarray | None = None) -> Analysis:
    token = token or CancelToken()
    t = {}
    t0 = time.perf_counter()
    if rgb is None:
        rgb = read_image(path).rgb                                       # P1
    t["read"] = time.perf_counter() - t0
    token.check()

    t0 = time.perf_counter()
    face = hub.get("face_landmarks", FaceLandmarks).detect(rgb)          # P2
    if face is None:
        raise FaceNotFound("얼굴을 찾지 못했습니다")
    missing = [k for k in REQUIRED_POINTS if k not in face.points]
    if missing:
        raise FaceNotFound(f"얼굴 점이 부족합니다: {missing}")
    t["face"] = time.perf_counter() - t0
    token.check()

    t0 = time.perf_counter()
    seg = hub.get("person_segmentation", PersonSegmentation).segment(rgb)
    t["segment"] = time.perf_counter() - t0
    token.check()

    t0 = time.perf_counter()                                             # P3 (0단계 범위)
    person = basic.refine_mask(rgb, seg.person)
    hair = basic.refine_mask(rgb, seg.hair) if seg.hair is not None else None
    t["refine"] = time.perf_counter() - t0

    skin = basic.clean_confidence(seg.face_skin) if seg.face_skin is not None else None
    a = Analysis(image=rgb, points=dict(face.points), person_mask=person, hair_mask=hair,
                 confidence={"face": face.confidence}, skin_mask=skin, regions=dict(face.regions))
    if seg.body_skin is not None:
        a.masks["몸피부"] = basic.clean_confidence(seg.body_skin)
    if seg.clothes is not None:
        a.masks["옷"] = basic.clean_confidence(seg.clothes)
    t0 = time.perf_counter()                                             # P3: 눈 꼭짓점·밑라인
    ey = eyes.measure_eyes(rgb, a.points)
    a.points.update(ey["points"])
    a.extra["level"] = ey["level"]
    a.points.update(nose_mouth.measure(rgb, a.regions, a.points))      # 코·입 중심 (사진에서)
    a.extra["sharpness"] = sharpness.feature_sharpness(rgb, a.points, a.regions)
    a.extra["catchlight"] = catchlight.judge(rgb, a.points)
    t["eyes"] = time.perf_counter() - t0
    token.check()

    t0 = time.perf_counter()                                             # P4 (규칙 방식): 남길 점
    a.extra["spots"] = spots.find_spots(rgb, a.points, a.regions, skin)
    kept = [sp for sp in a.extra["spots"] if sp.keep]
    a.regions["보존점"] = np.array([[sp.x, sp.y] for sp in kept], np.float32).reshape(-1, 2)
    a.extra["보존점_r"] = [sp.r for sp in kept]
    t["spots"] = time.perf_counter() - t0
    a.extra["timings"] = t
    a.extra["face_count"] = face.extra.get("faces", 1)
    log.info("prepare %s", {k: round(v * 1000) for k, v in t.items()})
    return a

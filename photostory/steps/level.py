"""1. 수평 (가이드 §4). 눈 바깥 꼭짓점 기준, 0.1° 미만은 생략.

꼭짓점은 prepare(P3, core/measure/eyes.py)에서 원본 해상도로 다시 찾은 점을 쓴다.
꼭짓점이 애매하면(주름·처진 눈꺼풀·흐린 초점) 양쪽 아랫눈꺼풀 밑라인으로 잡는다.
어떤 기준을 썼는지는 결과 화면 메모에 남긴다.
"""
from __future__ import annotations

import cv2

from ..core.measure.basic import tilt_degrees
from ..core.measure.eyes import angle
from ..core.pipeline import Frame, Step

MIN_DEG = 0.1


class LevelStep(Step):
    id = "level"
    title = "수평 맞추기"
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        p = frame.points
        info = (analysis.extra.get("level", {}) if analysis is not None else {})
        basis = info.get("basis", "꼭짓점(모델 점)")
        if basis == "밑라인" and "왼눈_밑라인" in p and "오른눈_밑라인" in p:
            deg = angle(p["왼눈_밑라인"], p["오른눈_밑라인"])
        else:
            deg = tilt_degrees(p)
        frame.outputs["tilt_deg"] = round(deg, 2)
        frame.outputs["level_basis"] = basis
        if basis != "꼭짓점":
            frame.notes.append(f"수평 기준: {basis} (눈 꼭짓점이 또렷하지 않음)")
        if abs(deg) < MIN_DEG:
            frame.notes.append(f"수평: {deg:.2f}° (0.1° 미만이라 생략)")
            return frame
        h, w = frame.image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)  # 반대로 돌려 수평을 맞춤
        out = frame.warp(M)
        out.notes.append(f"수평: {deg:.2f}° 회전")
        return out

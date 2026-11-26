"""1. 수평 (가이드 §4). 눈 바깥 꼭짓점 기준, 0.1° 미만은 생략.

0단계는 모델 점을 그대로 쓴다. 가이드는 모델 눈꼬리 점이 실제 꼭짓점보다 아래·안쪽에 찍힌다고
경고하므로, 1단계 정밀 측정(P3)에서 꼭짓점을 다시 찾고 아랫눈꺼풀 대체 기준을 추가한다.
"""
from __future__ import annotations

import cv2

from ..core.measure.basic import tilt_degrees
from ..core.pipeline import Frame, Step

MIN_DEG = 0.1


class LevelStep(Step):
    id = "level"
    title = "수평 맞추기"
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        deg = tilt_degrees(frame.points)
        frame.outputs["tilt_deg"] = round(deg, 2)
        if abs(deg) < MIN_DEG:
            frame.notes.append(f"수평: {deg:.2f}° (0.1° 미만이라 생략)")
            return frame
        h, w = frame.image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)  # 반대로 돌려 수평을 맞춤
        out = frame.warp(M)
        out.notes.append(f"수평: {deg:.2f}° 회전")
        return out

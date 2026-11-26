"""2. 안내선 계산 (가이드 §5 일부, §15-2). 0단계는 전송 배치에 필요한 값만 계산한다.

남녀 템플릿 전체(§5-1·§5-2)는 1단계 정밀 측정에서 추가한다.
"""
from __future__ import annotations

from ..core.measure.basic import center_x, crown_y, face_width
from ..core.pipeline import Frame, Step


class GuidesStep(Step):
    id = "guides"
    title = "안내선 계산"
    optional = False

    def run(self, frame: Frame, analysis, params) -> Frame:
        p = frame.points
        cx = center_x(p)
        chin = p["턱끝"][1]
        crown = None
        if frame.person_mask is not None:
            crown = crown_y(frame.person_mask, cx, 0.08 * face_width(p))
        if crown is None or crown >= p["이마_위"][1]:
            # 사람 영역이 이상하면 이마 위 점에서 얼굴 길이로 추정 (보수적 기본값)
            crown = p["이마_위"][1] - 0.35 * (chin - p["이마_위"][1])
            frame.notes.append("정수리: 사람 영역으로 찾지 못해 얼굴 비율로 추정")
        eye_y = (p["왼눈_바깥꼭짓점"][1] + p["오른눈_바깥꼭짓점"][1]) / 2
        frame.outputs["guides"] = {
            "center_x": cx, "crown_y": float(crown), "chin_y": float(chin), "eye_y": eye_y,
        }
        return frame

"""2. 안내선 계산 (가이드 §5, §15-2).

- 전송 배치에 쓰는 값: 중심선, 정수리, 턱, 눈 높이 (outputs["guides"])
- 남녀 템플릿 안내선 (§5-1·§5-2, core/measure/template.py): outputs["template"]
"""
from __future__ import annotations

from ..core.measure.basic import center_x, crown_y, face_width
from ..core.measure.template import measure_template
from ..core.pipeline import Frame, Step


class GuidesStep(Step):
    id = "guides"
    title = "안내선 계산"
    optional = False
    param_keys = ("gender",)

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
        try:
            t = measure_template(frame, params.get("gender", "여"))
            frame.outputs["template"] = t
            frame.notes += [f"안내선: {n}" for n in t.notes]
        except Exception as e:  # 안내선은 표시·측정용이라 실패해도 출력은 계속한다
            frame.notes.append(f"안내선: 템플릿 측정 실패 ({e})")
        return frame

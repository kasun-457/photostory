"""2단계: 목라인 좌우대칭 (가이드 §10). 얼굴·목·어깨 위치를 아는 합성 Frame으로 검사."""
import math

import cv2
import numpy as np

from photostory.core.measure.template import Line, Template
from photostory.core.pipeline import Frame, RenderParams
from photostory.steps.neck import NeckStep, decide, measure_neck, shoulder_tilt

H, W = 700, 600
CX = 300


def _frame(neck_off=0.0, tilt_deg=0.0, bend_basis="영상"):
    img = np.full((H, W, 3), 230, np.uint8)
    face = np.zeros((H, W), np.float32)
    cv2.ellipse(face, (CX, 260), (115, 150), 0, 0, 360, 1.0, -1)
    neck = np.zeros((H, W), np.float32)
    x0 = int(CX - 60 + neck_off)
    neck[360:560, x0:x0 + 120] = 1.0
    clothes = np.zeros((H, W), np.float32)
    xs = np.arange(W)
    top = (470 + np.tan(np.radians(tilt_deg)) * (xs - CX)).astype(int)
    for x in range(W):
        clothes[top[x]:, x] = 1.0
    clothes[:, x0:x0 + 120] = np.where(np.arange(H)[:, None] < 520, 0, clothes[:, x0:x0 + 120])
    img[clothes > 0] = (40, 40, 60)
    img[(neck > 0) | (face > 0)] = (200, 170, 150)
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    oval = np.stack([CX + 115 * np.cos(t), 260 + 150 * np.sin(t)], axis=1).astype(np.float32)
    pts = {"왼눈_바깥꼭짓점": (240, 120), "오른눈_바깥꼭짓점": (360, 120),
           "왼눈_안쪽꼭짓점": (280, 120), "오른눈_안쪽꼭짓점": (320, 120),
           "왼볼_입높이": (190, 300), "오른볼_입높이": (410, 300)}
    f = Frame(img, pts, np.clip(face + neck + clothes, 0, 1), None, 1.0, skin_mask=face,
              regions={"얼굴윤곽": oval}, masks={"몸피부": neck, "옷": clothes})
    tpl = Template("여", CX)
    tpl.h += [Line("I", "턱·목 꺾임", 380.0, bend_basis), Line("J", "턱 끝", 410.0, "영상")]
    f.outputs["template"] = tpl
    return f


def _params(st=1.0):
    return RenderParams({"strength.neck": st, "gender": "여"})


def test_measure_neck_and_shoulder():
    f = _frame(neck_off=10, tilt_deg=-3)
    union = np.clip(f.skin_mask + f.masks["몸피부"], 0, 1)
    n = measure_neck(union, CX, 387, 420, 220)
    assert abs(n.offset - 10) < 1.5 and abs(n.width - 119) < 3
    t = shoulder_tilt(f.masks["옷"], CX, 410, 220)
    assert abs(t - (-3)) < 0.5


def test_decide_rules():
    assert decide(2.0, 1.0, 200, 0.0, 1.0) == (None, 0.0)              # 3px 미만: 안 함
    m, v = decide(16.0, 8.0, 200, 0.0, 1.0)
    assert m == "shift" and v == -8.0                                  # 20px 미만: 유동화
    m, v = decide(24.0, 12.0, 260, -3.0, 1.0)
    assert m == "rotate" and abs(v - math.degrees(math.atan2(12, 260))) < 1e-6
    m, v = decide(24.0, 12.0, 260, 0.0, 1.0)
    assert m == "rotate" and abs(v - math.degrees(math.atan2(12, 260)) / 2) < 1e-6   # 어깨 → 절반
    assert decide(60.0, 30.0, 100, 0.0, 1.0) == (None, 0.0)            # 절반도 어깨가 기울면 안 함


def test_small_neck_offset_shifts_body():
    f = NeckStep().run(_frame(neck_off=8), None, _params())
    n = f.outputs["neck"]
    assert n["method"] == "shift" and abs(n["before"] - 16) < 3
    assert abs(n["after"]) < 0.25 * abs(n["before"]), n               # 턱 바로 아래는 조금 덜 움직임
    assert abs(n["tilt_after"] - n["tilt_before"]) < 0.3                 # 어깨 기울기 그대로
    assert abs(n["width_after"] - n["width_before"]) < 3                  # 목 폭 그대로


def test_big_offset_rotates_and_levels_shoulders():
    f = NeckStep().run(_frame(neck_off=12, tilt_deg=-3), None, _params())
    n = f.outputs["neck"]
    assert n["method"] == "rotate" and n["value"] > 0, n
    assert abs(n["after"]) < 0.4 * abs(n["before"]), n
    assert abs(n["tilt_after"]) < abs(n["tilt_before"]), n              # 어깨도 수평 쪽으로
    assert abs(n["width_after"] - n["width_before"]) < 4, n


def test_face_does_not_move():
    f0 = _frame(neck_off=12, tilt_deg=-3)
    before = f0.image.copy()
    f = NeckStep().run(f0, None, _params())
    assert np.abs(f.image[100:330].astype(int) - before[100:330]).max() <= 2


def test_skip_when_bend_estimated_or_strength_zero():
    f = NeckStep().run(_frame(neck_off=12, bend_basis="추정"), None, _params())
    assert "neck" not in f.outputs and any("꺾임" in n for n in f.notes)
    f = NeckStep().run(_frame(neck_off=12, tilt_deg=-3), None, _params(0.0))
    assert f.outputs["neck"]["method"] is None

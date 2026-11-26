"""2단계: 어깨 좌우 밸런스 (가이드 §14). 어깨 높이를 아는 합성 Frame으로 검사."""
import cv2
import numpy as np
import pytest

from photostory.core.pipeline import Frame, RenderParams
from photostory.core.session import Placement
from photostory.core.specs import SpecRegistry
from photostory.steps.shoulder import ShoulderStep
from photostory.steps.transfer import crop_rect

H, W = 1000, 800
CX, CROWN, CHIN = 400.0, 120.0, 420.0
REG = SpecRegistry()


def _frame(left_up=0.0, hair_on_left=False, specs=("half",)):
    """어깨선: 중심에서 바깥으로 내려가는 선. 왼쪽 어깨를 left_up px 올린다."""
    img = np.full((H, W, 3), 255, np.uint8)
    f = Frame(img, {"왼볼_입높이": (CX - 110, 330), "오른볼_입높이": (CX + 110, 330),
                    "왼눈_바깥꼭짓점": (CX - 70, 250), "오른눈_바깥꼭짓점": (CX + 70, 250)},
              None, None, 1.0)
    f.outputs["guides"] = {"center_x": CX, "crown_y": CROWN, "chin_y": CHIN, "eye_y": 250.0}
    # 테두리가 어깨를 지나도록 기울기를 맞춘다: 가장 넓은 규격의 변에서 (턱 + 아래 변)/2 높이
    rects = [crop_rect(f, REG.get(s), Placement()) for s in specs]
    x0, _, x1, y1, _ = min(rects, key=lambda r: r[0])
    yc = CHIN + 40
    k = ((CHIN + y1) / 2 - yc) / ((x1 - x0) / 2)
    person = np.zeros((H, W), np.float32)
    cv2.ellipse(person, (int(CX), int((CROWN + CHIN) / 2)), (120, int((CHIN - CROWN) / 2)), 0, 0, 360, 1.0, -1)
    person[int(CHIN) - 20:, int(CX) - 60:int(CX) + 60] = 1.0
    xs = np.arange(W)
    top = yc + k * np.abs(xs - CX) - np.where(xs < CX, left_up * np.clip((CX - xs) / 120, 0, 1), 0)
    for x in range(W):
        person[int(round(top[x])):, x] = 1.0
    person = cv2.GaussianBlur(person, (0, 0), 1.0)
    img[person > 0.5] = (60, 70, 120)
    hair = None
    if hair_on_left:
        hair = np.zeros((H, W), np.float32)
        hair[int(CROWN):int(y1), :int(CX) - 60] = person[int(CROWN):int(y1), :int(CX) - 60]
    f.person_mask, f.hair_mask = person, hair
    return f


def _params(specs=("half",), st=1.0):
    return RenderParams({"strength.neck": st, "specs": [REG.get(s) for s in specs],
                         "placements": {s: Placement() for s in specs}})


def test_balanced_shoulders_untouched():
    f = _frame(0.0)
    before = f.image.copy()
    out = ShoulderStep().run(f, None, _params())
    assert abs(out.outputs["shoulder"]["half"]["before"]) < 1.0
    assert np.array_equal(out.image, before)


def test_high_shoulder_lowered():
    f = ShoulderStep().run(_frame(left_up=12.0), None, _params())
    r = f.outputs["shoulder"]["half"]
    assert r["before"] < -3, r                         # 왼쪽이 높음
    assert abs(r["after"]) <= 1.0, r                  # 전송본 1px 이내
    assert any("왼쪽 어깨" in n for n in f.notes)


def test_multiple_specs():
    specs = ("half", "passport")
    f = ShoulderStep().run(_frame(left_up=12.0, specs=specs), None, _params(specs))
    for sid in specs:
        r = f.outputs["shoulder"][sid]
        assert "skip" not in r, r
        assert abs(r["after"]) <= 1.5, (sid, r)


@pytest.mark.parametrize("case", ["hair", "huge", "zero"])
def test_skips(case):
    if case == "hair":
        f = ShoulderStep().run(_frame(left_up=12.0, hair_on_left=True), None, _params())
        assert "머리카락" in f.outputs["shoulder"]["half"]["skip"]
    elif case == "huge":
        f = ShoulderStep().run(_frame(left_up=60.0), None, _params())
        assert "너무 큼" in f.outputs["shoulder"]["half"]["skip"]
    else:
        f0 = _frame(left_up=12.0)
        before = f0.image.copy()
        f = ShoulderStep().run(f0, None, _params(st=0.0))
        assert np.array_equal(f.image, before)

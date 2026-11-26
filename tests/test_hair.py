"""2단계: 머리카락 (가이드 §11 잔머리, §13 밝기). 합성 Frame으로 검사."""
import cv2
import numpy as np

from photostory.core.pipeline import Frame, RenderParams
from photostory.steps.background import BackgroundStep
from photostory.steps.hair import HairBrightnessStep, HairTidyStep

H, W = 500, 520
CX = 260


def _frame(hair_rgb=(70, 55, 40), strands=True):
    img = np.full((H, W, 3), 245, np.uint8)
    person = np.zeros((H, W), np.float32)
    cv2.circle(person, (CX, 200), 110, 1.0, -1)
    hair = person.copy()
    face = np.zeros((H, W), np.float32)
    cv2.ellipse(face, (CX, 235), (80, 95), 0, 0, 360, 1.0, -1)
    if strands:
        for ang in (-60, -20, 25):                      # 머리 위로 삐친 가는 올 3개
            a = np.radians(ang - 90)
            p0 = (int(CX + 108 * np.cos(a)), int(200 + 108 * np.sin(a)))
            p1 = (int(CX + 150 * np.cos(a)), int(200 + 150 * np.sin(a)))
            for m in (person, hair):
                cv2.line(m, p0, p1, 1.0, 2)
        for m in (person, hair):                        # 굵은 머리 다발 (지우면 안 됨)
            cv2.rectangle(m, (CX + 95, 120), (CX + 140, 170), 1.0, -1)
    hair[face > 0] = 0
    img[person > 0.5] = hair_rgb
    img[face > 0] = (215, 180, 160)
    person[300:, CX - 50:CX + 50] = 1.0                 # 목
    img[300:, CX - 50:CX + 50] = (215, 180, 160)
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    oval = np.stack([CX + 80 * np.cos(t), 235 + 95 * np.sin(t)], axis=1).astype(np.float32)
    f = Frame(img, {"왼볼_입높이": (CX - 80, 280), "오른볼_입높이": (CX + 80, 280)}, person, hair, 1.0,
              skin_mask=face, regions={"얼굴윤곽": oval})
    f.outputs["guides"] = {"center_x": float(CX), "crown_y": 90.0, "chin_y": 330.0, "eye_y": 210.0}
    return f


def _p(st=1.0):
    return RenderParams({"strength.hair": st, "bg_color": (255, 255, 255)})


def _strand_point(ang):
    a = np.radians(ang - 90)
    return int(CX + 130 * np.cos(a)), int(200 + 130 * np.sin(a))


def test_tidy_removes_thin_strands_keeps_rest():
    f = HairTidyStep().run(_frame(), None, _p())
    m = f.person_mask
    for ang in (-60, -20, 25):
        x, y = _strand_point(ang)
        assert m[y - 2:y + 3, x - 2:x + 3].max() < 0.05, ang
    assert m[145, CX + 120] > 0.95                      # 굵은 다발
    assert m[200, CX] > 0.99 and m[200, CX - 105] > 0.9  # 머리 안·윤곽 안쪽
    f0 = _frame(strands=False)
    edge = (f0.person_mask > 0.05) & (f0.person_mask < 0.95)
    f1 = HairTidyStep().run(_frame(strands=False), None, _p())
    assert "hair_tidy" not in f1.outputs                 # 올이 없으면 아무것도 안 함
    assert np.array_equal(f1.person_mask[edge], f0.person_mask[edge])


def test_tidy_then_background_paints_strands():
    f = HairTidyStep().run(_frame(), None, _p())
    f = BackgroundStep().run(f, None, _p())
    for ang in (-60, -20, 25):
        x, y = _strand_point(ang)
        assert (f.image[y, x] == 255).all(), ang


def test_tidy_strength_zero():
    f0 = _frame()
    f = HairTidyStep().run(_frame(), None, _p(0.0))
    assert np.array_equal(f.person_mask, f0.person_mask)


def test_bright_hair_darkened_dark_hair_kept():
    f = HairBrightnessStep().run(_frame(hair_rgb=(235, 225, 205), strands=False), None, _p())
    r = f.outputs["hair_bright"]
    assert r["applied"] and r["median"] > 185 and 150 <= r["after"] <= 180, r
    assert (f.image[235, CX] == (215, 180, 160)).all()   # 얼굴 피부는 그대로
    f2 = HairBrightnessStep().run(_frame(hair_rgb=(70, 55, 40), strands=False), None, _p())
    assert not f2.outputs["hair_bright"]["applied"]
    f3 = HairBrightnessStep().run(_frame(hair_rgb=(235, 225, 205), strands=False), None, _p(0.0))
    assert "hair_bright" not in f3.outputs

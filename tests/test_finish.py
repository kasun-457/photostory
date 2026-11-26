"""2단계: 마무리 유동화 (가이드 §9-5). 눈썹·볼·머리 위치를 아는 합성 Frame으로 검사."""
import cv2
import numpy as np

from photostory.core.pipeline import Frame, RenderParams
from photostory.steps.finish import FinishStep, brow_bottoms, head_rows

H, W = 700, 640
CX = 320
FACE_W = 230          # 입 높이 얼굴 폭 (왼볼~오른볼)


def _frame(brow_dy=0, cheek_dx=0, hair_dx=0, hair_on_cheek=False):
    """brow_dy: 사진 왼쪽 눈썹을 이만큼 아래로. cheek_dx: 콧볼 높이 왼쪽 볼을 이만큼 바깥으로.
    hair_dx: 머리 왼쪽 바깥을 이만큼 넓게."""
    img = np.full((H, W, 3), 245, np.uint8)
    hair = np.zeros((H, W), np.float32)
    cv2.ellipse(hair, (CX, 230), (175, 190), 0, 180, 360, 1.0, -1)
    cv2.rectangle(hair, (CX - 175, 230), (CX + 175, 300), 1.0, -1)
    if hair_dx:
        cv2.ellipse(hair, (CX - hair_dx // 2, 230), (175 + hair_dx // 2, 190), 0, 180, 270, 1.0, -1)
        cv2.rectangle(hair, (CX - 175 - hair_dx, 230), (CX, 300), 1.0, -1)
    face = np.zeros((H, W), np.float32)
    cv2.ellipse(face, (CX, 320), (115, 160), 0, 0, 360, 1.0, -1)
    if cheek_dx:
        cv2.ellipse(face, (CX - 115 + 10, 345), (10 + cheek_dx, 50), 0, 0, 360, 1.0, -1)
    if hair_on_cheek:
        hair[300:420, CX - 160:CX - 110] = 1.0
    hair[face > 0] = 0
    img[hair > 0.5] = (60, 45, 35)
    img[face > 0] = (215, 180, 160)
    # 눈썹: 어두운 띠 (가운데 위·아래가 피부색이라 아래 끝 경계가 뚜렷함)
    brows = {}
    for key, x0, dy in (("왼눈썹", CX - 95, brow_dy), ("오른눈썹", CX + 25, 0)):
        y0 = 215 + dy
        img[y0:y0 + 14, x0:x0 + 70] = (70, 55, 45)
        brows[key] = np.float32([[x0, y0], [x0 + 70, y0], [x0 + 70, y0 + 14], [x0, y0 + 14]])
    eyes = {}
    for key, x0 in (("왼눈", CX - 90), ("오른눈", CX + 30)):
        img[262:276, x0:x0 + 60] = (240, 240, 240)
        eyes[key] = np.float32([[x0, 262], [x0 + 60, 262], [x0 + 60, 276], [x0, 276]])
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    oval = np.stack([CX + 115 * np.cos(t), 320 + 160 * np.sin(t)], axis=1).astype(np.float32)
    pts = {"왼눈_바깥꼭짓점": (CX - 90, 269), "오른눈_바깥꼭짓점": (CX + 90, 269),
           "왼눈_안쪽꼭짓점": (CX - 30, 269), "오른눈_안쪽꼭짓점": (CX + 30, 269),
           "왼콧볼": (CX - 25, 345), "오른콧볼": (CX + 25, 345),
           "왼입꼬리": (CX - 35, 395), "오른입꼬리": (CX + 35, 395),
           "왼볼_입높이": (CX - FACE_W / 2, 395), "오른볼_입높이": (CX + FACE_W / 2, 395)}
    person = np.clip(hair + face, 0, 1)
    f = Frame(img, pts, person, hair, 1.0, skin_mask=face,
              regions={"얼굴윤곽": oval, **brows, **eyes}, masks={})
    f.outputs["guides"] = {"center_x": float(CX), "crown_y": 40.0, "chin_y": 480.0, "eye_y": 269.0}
    return f


def _p(**st):
    return RenderParams({f"strength.{k}": v for k, v in {"brows": 1.0, "jaw": 1.0, "hair": 1.0, **st}.items()})


def _gray(f):
    return cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _brow_diff(f):
    b = brow_bottoms(_gray(f), f.regions, FACE_W)
    return b["왼눈썹"] - b["오른눈썹"]


def _cheek(f):
    row = f.skin_mask[345] > 0.5
    xs = np.flatnonzero(row)
    return (CX - xs[0]) - (xs[-1] - CX)


def _hair_diff(f):
    head = np.maximum(f.hair_mask, f.skin_mask)
    ys, L, Rr = head_rows(head, CX, 80, 260, 1)
    return float(np.median((CX - L) - (Rr - CX)))


def test_symmetric_face_untouched():
    f0 = _frame()
    img = f0.image.copy()
    f = FinishStep().run(f0, None, _p())
    assert f.outputs["finish"]["moved"] == [] and np.array_equal(f.image, img)


def test_brows_meet_halfway_eyes_stay():
    f0 = _frame(brow_dy=8)
    assert abs(_brow_diff(f0) - 8) <= 1
    eyes_before = f0.image[262:276].copy()
    f = FinishStep().run(f0, None, _p())
    assert any("눈썹" in m for m in f.outputs["finish"]["moved"]), f.outputs
    assert abs(_brow_diff(f)) <= 1.5, _brow_diff(f)
    assert np.abs(f.image[262:276].astype(int) - eyes_before).max() <= 3        # 눈은 그대로


def test_cheek_wide_side_pulled_in():
    f0 = _frame(cheek_dx=12)
    before = _cheek(f0)
    assert before >= 10
    f = FinishStep().run(f0, None, _p())
    assert abs(_cheek(f)) <= 0.3 * before, (_cheek(f), before)
    # 입 높이 아래(대칭 단계가 맡는 곳)는 그대로
    assert np.abs(f.skin_mask[420:].astype(np.float32) - f0.skin_mask[420:]).max() < 0.05


def test_cheek_skipped_when_hair_touches():
    f = FinishStep().run(_frame(cheek_dx=12, hair_on_cheek=True), None, _p())
    assert not any("볼" in m for m in f.outputs["finish"]["moved"])


def test_hair_outline_balanced_and_style_kept():
    f0 = _frame(hair_dx=12)
    before = _hair_diff(f0)
    assert before >= 8
    f = FinishStep().run(f0, None, _p())
    assert any("머리" in m for m in f.outputs["finish"]["moved"]), f.outputs
    assert abs(_hair_diff(f)) <= 0.4 * before, (_hair_diff(f), before)
    # 머리 모양만큼 크게 다르면(얼굴 폭 6% 넘게) 그대로 둔다
    g = FinishStep().run(_frame(hair_dx=40), None, _p())
    assert not any("머리" in m for m in g.outputs["finish"]["moved"])


def test_strength_zero_does_nothing():
    f = FinishStep().run(_frame(brow_dy=8, cheek_dx=12, hair_dx=12), None, _p(brows=0, jaw=0, hair=0))
    assert "finish" not in f.outputs

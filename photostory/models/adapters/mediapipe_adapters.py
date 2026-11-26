"""MediaPipe 어댑터 (얼굴점 478개, 셀피 다중 분할, 자세).

모델 번호 → 표준 이름 변환은 이 파일 안에서만 한다.
MediaPipe 얼굴 그물망 번호는 '손님 기준' 좌우라서, 사진 기준으로 바꿔 적는다.
(33 = 손님 오른눈 바깥 = 사진 왼쪽)
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from ..roles import FaceLandmarks, FaceResult, PersonSegmentation, PoseLandmarks, PoseResult, SegmentResult

# 모델은 긴 변이 이 정도일 때 충분하다. 더 크면 줄여서 넣는다 (정밀 측정은 P3에서).
_DETECT_MAX_SIDE = 1600

FACE_MESH_INDEX = {
    "왼눈_바깥꼭짓점": 33, "오른눈_바깥꼭짓점": 263,
    "왼눈_안쪽꼭짓점": 133, "오른눈_안쪽꼭짓점": 362,
    "왼눈_홍채중심": 468, "오른눈_홍채중심": 473,
    "왼눈_윗꺼풀": 159, "오른눈_윗꺼풀": 386,
    "왼눈_아랫꺼풀": 145, "오른눈_아랫꺼풀": 374,
    "왼눈썹_바깥": 70, "오른눈썹_바깥": 300,
    "왼눈썹_안쪽": 107, "오른눈썹_안쪽": 336,
    "왼눈썹_위": 105, "오른눈썹_위": 334,
    "코끝": 1, "코밑": 2, "왼콧볼": 129, "오른콧볼": 358,
    "왼입꼬리": 61, "오른입꼬리": 291, "윗입술_중심": 0, "아랫입술_중심": 17,
    "왼볼_입높이": 132, "오른볼_입높이": 361,
    "왼턱선": 172, "오른턱선": 397,
    "왼턱끝쪽": 149, "오른턱끝쪽": 378,
    "왼얼굴_옆": 234, "오른얼굴_옆": 454,
    "턱끝": 152, "이마_위": 10,
    "윗입술_안쪽": 13, "아랫입술_안쪽": 14,
}

# 홍채 둘레 4점 (중심 468 → 469~472, 중심 473 → 474~477). 위·아래·좌우는 좌표로 가른다.
IRIS_RING = {"왼눈": (468, [469, 470, 471, 472]), "오른눈": (473, [474, 475, 476, 477])}


def _iris_points(pts: np.ndarray) -> dict[str, tuple[float, float]]:
    out = {}
    if len(pts) < 478:
        return out
    for eye, (_, ring) in IRIS_RING.items():
        r = pts[ring]
        top, bottom = r[np.argmin(r[:, 1])], r[np.argmax(r[:, 1])]
        left, right = r[np.argmin(r[:, 0])], r[np.argmax(r[:, 0])]
        inner, outer = (right, left) if eye == "왼눈" else (left, right)   # 코 쪽이 안쪽
        for k, v in (("홍채위", top), ("홍채아래", bottom), ("홍채안쪽", inner), ("홍채바깥", outer)):
            out[f"{eye}_{k}"] = (float(v[0]), float(v[1]))
    return out


# 얼굴 부위 영역. MediaPipe 이름의 RIGHT_* 는 손님 기준이라 사진 왼쪽이다.
FACE_REGION_INDEX = {
    "얼굴윤곽": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
                54, 103, 67, 109],
    "왼눈": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "오른눈": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "왼눈썹": [46, 53, 52, 65, 55, 107, 66, 105, 63, 70],
    "오른눈썹": [276, 283, 282, 295, 285, 336, 296, 334, 293, 300],
    "입술": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37,
            39, 40, 185],
    "코아래": [129, 49, 64, 98, 97, 2, 326, 327, 294, 279, 358],
    "콧대": [168, 6, 197, 195, 5, 4],
    "왼턱선_선": [93, 132, 58, 172, 136, 150, 149, 176, 148],
    "오른턱선_선": [323, 361, 288, 397, 365, 379, 378, 400, 377],
}

# MediaPipe 자세: 11 = 왼어깨(손님 기준) = 사진 오른쪽
POSE_INDEX = {"왼어깨": 12, "오른어깨": 11, "코_자세": 0}


def _shrink(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb), s


class _MPBase:
    def __init__(self, config: dict):
        self.config = config
        self._task = None
        self._lock = threading.Lock()   # MediaPipe 작업 객체는 한 번에 하나씩

    def _create(self):  # pragma: no cover
        raise NotImplementedError

    def load(self) -> None:
        with self._lock:
            if self._task is None:
                self._task = self._create()

    def close(self) -> None:
        with self._lock:
            if self._task is not None:
                try:
                    self._task.close()
                except Exception:
                    pass
                self._task = None

    def _run(self, fn):
        self.load()
        with self._lock:
            return fn(self._task)

    def _base_options(self):
        # 경로 대신 내용을 넘긴다. Windows의 MediaPipe는 한글 등이 든 경로의 파일을 열지 못한다
        # (예: C:\사진이야기\... → "Unable to open file", 2026-11-26 확인).
        from pathlib import Path
        from mediapipe.tasks.python import BaseOptions
        return BaseOptions(model_asset_buffer=Path(self.config["path"]).read_bytes())

    @staticmethod
    def _mp_image(rgb):
        import mediapipe as mp
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


class MediaPipeFace(_MPBase, FaceLandmarks):
    def _create(self):
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=self._base_options(),
            num_faces=2, min_face_detection_confidence=0.4)
        return vision.FaceLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> FaceResult | None:
        small, s = _shrink(rgb, _DETECT_MAX_SIDE)
        res = self._run(lambda t: t.detect(self._mp_image(small)))
        if not res.face_landmarks:
            return None
        h, w = small.shape[:2]
        # 여러 얼굴이면 가장 큰 얼굴
        faces = [np.array([[p.x * w, p.y * h] for p in lm], np.float64) for lm in res.face_landmarks]
        pts = max(faces, key=lambda a: np.ptp(a[:, 0]) * np.ptp(a[:, 1])) / s
        named = {k: (float(pts[i, 0]), float(pts[i, 1])) for k, i in FACE_MESH_INDEX.items() if i < len(pts)}
        named.update(_iris_points(pts))
        regions = {k: pts[idx].astype(np.float32) for k, idx in FACE_REGION_INDEX.items()
                   if max(idx) < len(pts)}
        return FaceResult(named, 1.0, pts, {"faces": len(faces)}, regions)


class MediaPipeSelfieMulticlass(_MPBase, PersonSegmentation):
    """클래스: 0 배경, 1 머리카락, 2 몸 피부, 3 얼굴 피부, 4 옷, 5 그 밖(장신구)."""

    def _create(self):
        from mediapipe.tasks.python import vision
        opts = vision.ImageSegmenterOptions(
            base_options=self._base_options(),
            output_confidence_masks=True, output_category_mask=False)
        return vision.ImageSegmenter.create_from_options(opts)

    def segment(self, rgb: np.ndarray) -> SegmentResult:
        small, s = _shrink(rgb, int(self.config.get("max_side", 1024)))
        res = self._run(lambda t: t.segment(self._mp_image(small)))
        ms = [np.array(m.numpy_view(), np.float32).reshape(small.shape[:2]) for m in res.confidence_masks]
        H, W = rgb.shape[:2]
        up = lambda m: cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) if s < 1 else m
        person = np.clip(1.0 - ms[0], 0, 1)
        return SegmentResult(person=up(person), hair=up(ms[1]), body_skin=up(ms[2]),
                             face_skin=up(ms[3]), clothes=up(ms[4]), accessories=up(ms[5]))


class MediaPipePose(_MPBase, PoseLandmarks):
    def _create(self):
        from mediapipe.tasks.python import vision
        opts = vision.PoseLandmarkerOptions(
            base_options=self._base_options(), num_poses=1)
        return vision.PoseLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> PoseResult | None:
        small, s = _shrink(rgb, 1024)
        res = self._run(lambda t: t.detect(self._mp_image(small)))
        if not res.pose_landmarks:
            return None
        h, w = small.shape[:2]
        lm = res.pose_landmarks[0]
        pts = {k: (lm[i].x * w / s, lm[i].y * h / s) for k, i in POSE_INDEX.items()}
        vis = min(lm[11].visibility or 0, lm[12].visibility or 0)
        return PoseResult(pts, float(vis))

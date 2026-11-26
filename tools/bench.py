"""원본 크기(4472×6708) 처리 시간 재기. 공개 테스트 사진(astronaut)을 키워서 쓴다.

사용: python tools/bench.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostory.core.pipeline import Pipeline  # noqa: E402
from photostory.core.prepare import analyze  # noqa: E402
from photostory.core.processor import Processor  # noqa: E402
from photostory.core.session import Customer, Session  # noqa: E402
from photostory.core.specs import SpecRegistry  # noqa: E402
from photostory.models.hub import ModelHub  # noqa: E402


def portrait(hub):
    from skimage import data
    img = cv2.resize(data.astronaut(), (1536, 1536), interpolation=cv2.INTER_CUBIC)
    m = hub.get("person_segmentation").segment(img).person
    m = cv2.GaussianBlur((m > 0.5).astype(np.float32), (0, 0), 2)[..., None]
    bg = np.empty_like(img)
    bg[:] = (244, 242, 238)
    comp = (img * m + bg * (1 - m)).astype(np.uint8)
    canvas = np.empty((2304, 1536, 3), np.uint8)
    canvas[:] = (244, 242, 238)
    canvas[400:400 + 1536] = comp
    return cv2.resize(canvas, (4472, 6708), interpolation=cv2.INTER_CUBIC)


def main():
    hub = ModelHub()
    img = portrait(hub)
    t0 = time.perf_counter()
    a = analyze("bench", hub, rgb=img)
    print(f"prepare {time.perf_counter() - t0:.2f}s")
    proc = Processor(hub, SpecRegistry())
    proc.pipeline.reset(a)
    s = Session(Customer(name="벤치", purposes=["passport", "half"]))
    for preview in (True, False):
        t0 = time.perf_counter()
        r = proc.render(s, preview)
        print(f"render({'미리보기' if preview else '원본'}) {time.perf_counter() - t0:.2f}s",
              {k: round(v, 2) for k, v in r.timings.items() if v > 0.05})
    for n in r.frame.notes:
        print("  메모:", n)
    hub.close()


if __name__ == "__main__":
    main()

"""테스트용 합성 안경 사진: 원본 얼굴에 안경테를 그리고 사진 전체를 조금 돌리고·키우고·옮긴다."""
import cv2
import numpy as np

from photostory.core.measure import regions as R


def draw_frame(img, pts, color=(25, 25, 30)):
    """눈 둘레 사각 테 2개 + 다리(브리지). 원본 좌표. 그린 테 마스크(0/255)를 돌려준다."""
    fw = R.face_w(pts)
    t = max(2, round(0.016 * fw))
    m = np.zeros(img.shape[:2], np.uint8)
    for e in ("왼눈", "오른눈"):
        o, i = pts[f"{e}_바깥꼭짓점"], pts[f"{e}_안쪽꼭짓점"]
        cx, cy = (o[0] + i[0]) / 2, (o[1] + i[1]) / 2
        hw, hh = 0.20 * fw, 0.13 * fw
        cv2.rectangle(m, (int(cx - hw), int(cy - hh)), (int(cx + hw), int(cy + hh)), 255, t)
    l, r = pts["왼눈_안쪽꼭짓점"], pts["오른눈_안쪽꼭짓점"]
    cy = (l[1] + r[1]) / 2 - 0.03 * fw
    cv2.line(m, (int((l[0] + r[0]) / 2 - 0.18 * fw), int(cy)), (int((l[0] + r[0]) / 2 + 0.18 * fw), int(cy)), 255, t)
    out = img.copy()
    out[m > 0] = color
    return out, m


def glasses_photo(portrait, pts, deg=1.5, scale=1.03, shift=(8, -5)):
    with_frame, m = draw_frame(portrait, pts)
    h, w = portrait.shape[:2]
    T = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    T[:, 2] += shift
    g = cv2.warpAffine(with_frame, T, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return g, m

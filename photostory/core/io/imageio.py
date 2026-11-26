"""사진 읽기·쓰기 (설계서 1-2, 4절).

- 한글 경로를 지원한다 (OpenCV의 imread 대신 Pillow·바이트 디코딩 사용).
- 읽을 때 EXIF 회전을 반영하고, 색 프로파일이 있으면 sRGB로 바꾼다.
- 모든 내부 이미지는 RGB uint8 (H, W, 3) numpy 배열이다.
- JPEG 저장은 Photoshop 품질 12에 가깝게(4:4:4, 높은 품질) + DPI + sRGB ICC.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".srw", ".pef"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUPPORTED_EXTS = IMAGE_EXTS | RAW_EXTS

# Photoshop 품질 12에 해당하는 값. 실제 Photoshop 결과와 비교해 맞춰야 한다 (설계서 8-4).
JPEG_QUALITY = 97

_SRGB = ImageCms.createProfile("sRGB")
_SRGB_BYTES = ImageCms.ImageCmsProfile(_SRGB).tobytes()


class ImageReadError(Exception):
    pass


@dataclass
class LoadedImage:
    rgb: np.ndarray          # (H, W, 3) uint8, sRGB
    path: Path
    dpi: tuple[float, float] | None = None


def is_supported(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTS


def _to_srgb(img: Image.Image) -> Image.Image:
    icc = img.info.get("icc_profile")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img = ImageCms.profileToProfile(img, src, _SRGB, outputMode="RGB")
        except (ImageCms.PyCMSError, OSError):
            pass  # 깨진 프로파일은 무시하고 그대로 쓴다
    return img


def _pil_to_rgb(img: Image.Image) -> np.ndarray:
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float32)
        arr = np.clip(arr / (256.0 if arr.max() > 255 else 1.0), 0, 255).astype(np.uint8)
        return np.repeat(arr[..., None], 3, axis=2)
    if img.mode == "RGBA":
        # 투명 부분은 흰색으로
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def _read_raw(path: Path) -> np.ndarray:
    try:
        import rawpy
    except ImportError as e:  # pragma: no cover - 설치 환경에 따라
        raise ImageReadError("RAW 파일을 열려면 rawpy가 필요합니다") from e
    with open(path, "rb") as f:  # 한글 경로 대응: 파일 객체로 연다
        with rawpy.imread(f) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB,
                                  output_bps=8, no_auto_bright=False)
    return np.ascontiguousarray(rgb)


def read_image(path: str | os.PathLike) -> LoadedImage:
    p = Path(path)
    if not p.exists():
        raise ImageReadError(f"파일이 없습니다: {p}")
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ImageReadError(f"지원하지 않는 형식입니다: {ext}")
    if ext in RAW_EXTS:
        return LoadedImage(_read_raw(p), p)
    try:
        with Image.open(p) as img:
            img.load()
            dpi = img.info.get("dpi")
            img = ImageOps.exif_transpose(img)
            img = _to_srgb(img)
            rgb = _pil_to_rgb(img)
    except (OSError, ValueError) as e:
        raise ImageReadError(f"사진을 열 수 없습니다: {p.name} ({e})") from e
    return LoadedImage(rgb, p, tuple(float(v) for v in dpi) if dpi else None)


def encode_jpeg(rgb: np.ndarray, dpi: float = 300, quality: int = JPEG_QUALITY) -> bytes:
    img = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, subsampling=0, optimize=True,
             dpi=(dpi, dpi), icc_profile=_SRGB_BYTES)
    return buf.getvalue()


def write_jpeg(path: str | os.PathLike, rgb: np.ndarray, dpi: float = 300,
               quality: int = JPEG_QUALITY) -> Path:
    """임시 파일에 쓴 뒤 이름을 바꿔서, 도중에 실패해도 기존 파일이 깨지지 않게 한다."""
    p = Path(path)
    data = encode_jpeg(rgb, dpi, quality)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)
    return p


def make_thumbnail(rgb: np.ndarray, max_side: int) -> np.ndarray:
    import cv2
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def read_thumbnail(path: str | os.PathLike, max_side: int = 600) -> np.ndarray:
    """업로드 칸 미리보기용 빠른 읽기. JPEG는 축소 디코딩(draft)을 써서 원본을 다 풀지 않는다."""
    p = Path(path)
    if p.suffix.lower() in RAW_EXTS:
        try:
            import rawpy
            with open(p, "rb") as f, rawpy.imread(f) as raw:
                th = raw.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                with Image.open(io.BytesIO(th.data)) as img:
                    img.draft("RGB", (max_side, max_side))
                    return make_thumbnail(_pil_to_rgb(ImageOps.exif_transpose(img)), max_side)
        except Exception:
            pass
        return make_thumbnail(read_image(p).rgb, max_side)
    try:
        with Image.open(p) as img:
            img.draft("RGB", (max_side, max_side))
            img = ImageOps.exif_transpose(img)
            img = _to_srgb(img)
            return make_thumbnail(_pil_to_rgb(img), max_side)
    except (OSError, ValueError) as e:
        raise ImageReadError(f"사진을 열 수 없습니다: {p.name} ({e})") from e

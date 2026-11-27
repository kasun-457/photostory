"""3-1 짝 목록: 원본 폴더를 읽기만 해서 원본·수정본 짝을 index.csv로 적는다 (학습설계 2).

사용:
  py -3.12 -m training.index_pairs                       # 기본: D:\\1_촬영\\3_증명사진 → C:\\photostory_data\\index.csv
  py -3.12 -m training.index_pairs --src 폴더 --work 폴더

- 원본 폴더에는 아무것도 쓰지 않는다. 파일은 머리(크기)만 읽는다 (safety.open_source).
- 학습/검증은 날짜로 나눈다: 가장 최근 VAL_MONTHS개월 = 검증. 같은 이름의 손님 폴더는 한쪽에만 둔다.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path

from PIL import Image

from .safety import DEFAULT_SOURCE, DEFAULT_WORK, check_output, open_source

VAL_MONTHS = 3
FOLDER_RE = re.compile(r"^(\d{8})_(.+)$")
ORIG_RE = re.compile(r"^(?!전송)(.+)_원본\.jpe?g$", re.IGNORECASE)
RETOUCH_SUFFIXES = ("_수정본.jpg", "_수정본.jpeg", "_수정.jpg")


@dataclass
class Pair:
    folder: str            # 원본 폴더 기준 상대 경로
    date: str              # YYYY-MM-DD (폴더 이름에서)
    name: str              # 손님 이름 (파일 이름 앞부분의 첫 토막). 학습/검증 나누기 묶음
    original: str          # 파일 이름
    retouched: str
    width: int = 0
    height: int = 0
    same_size: bool = False
    glasses: bool = False  # 폴더에 '안경' 파일이 있음 → 안경 얹기가 섞여 있어 따로 다룸
    psd: bool = False      # {X}_수정과정.psd 있음
    multi: bool = False    # 한 폴더에 짝이 여러 개 (여러 명 또는 용도별)
    tag: str = ""          # 폴더 이름 끝 표시 (x, ok). x = 고객에게 파일 보냈는지 표시일 뿐, 학습과 상관없음
    split: str = ""        # train / val
    status: str = "ok"     # ok / 수정본없음 / 크기다름 / 읽기실패


def _size(path: Path) -> tuple[int, int] | None:
    try:
        with open_source(path) as fh, Image.open(fh) as im:
            return im.size
    except OSError:
        return None


def scan_folder(src: Path, d: Path) -> list[Pair]:
    """손님 폴더 하나 → 짝 목록. 짝은 파일 이름으로: {X}_원본.jpg ↔ {X}_수정본.jpg (없으면 {X}_수정.jpg)."""
    m = FOLDER_RE.match(d.name)
    if not m:
        return []
    ds = m.group(1)
    try:
        day = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))
    except ValueError:
        return []
    last = m.group(2).rsplit("_", 1)[-1]
    tag = last.lower().strip(".") if last.lower().strip(".") in ("x", "ok") else ""
    files = [f.name for f in d.iterdir() if f.is_file()]
    low = {f.lower(): f for f in files}
    glasses = any("안경" in f for f in files)
    out = []
    for f in sorted(files):
        om = ORIG_RE.match(f)
        if not om:
            continue
        stem = om.group(1)
        r = next((low[(stem + suf).lower()] for suf in RETOUCH_SUFFIXES if (stem + suf).lower() in low), "")
        p = Pair(folder=str(d.relative_to(src)), date=day.isoformat(), name=stem.split("_")[0],
                 original=f, retouched=r, glasses=glasses,
                 psd=(stem + "_수정과정.psd").lower() in low, tag=tag)
        if not r:
            p.status = "수정본없음"
        else:
            so, sr = _size(d / f), _size(d / r)
            if so is None or sr is None:
                p.status = "읽기실패"
            else:
                p.width, p.height = so
                p.same_size = so == sr
                if not p.same_size:
                    p.status = "크기다름"
        out.append(p)
    if len(out) > 1:
        for p in out:
            p.multi = True
    return out


def assign_split(pairs: list[Pair], val_months: int = VAL_MONTHS) -> None:
    if not pairs:
        return
    last = max(date.fromisoformat(p.date) for p in pairs)
    y, mth = last.year, last.month - val_months
    while mth <= 0:
        y, mth = y - 1, mth + 12
    cut = date(y, mth, min(last.day, 28))
    val_names = {p.name for p in pairs if date.fromisoformat(p.date) > cut}
    for p in pairs:
        p.split = "val" if p.name in val_names else "train"


def customer_dirs(src: Path) -> list[Path]:
    # {년}/{월}/{손님}
    return sorted(d for d in src.glob("*/*/*") if d.is_dir())


def build_index(src: Path, workers: int = 8, progress=None) -> list[Pair]:
    dirs = customer_dirs(src)
    out: list[Pair] = []
    with ThreadPoolExecutor(workers) as ex:
        for i, ps in enumerate(ex.map(lambda d: scan_folder(src, d), dirs)):
            out.extend(ps)
            if progress and (i + 1) % 1000 == 0:
                progress(i + 1, len(dirs))
    assign_split(out)
    return out


def write_index(pairs: list[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=[f.name for f in fields(Pair)])
        w.writeheader()
        for p in pairs:
            w.writerow(asdict(p))


def read_index(path: Path) -> list[Pair]:
    ints, bools = {"width", "height"}, {"same_size", "glasses", "psd", "multi"}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return [Pair(**{k: (int(v) if k in ints else v == "True" if k in bools else v)
                    for k, v in r.items()}) for r in rows]


RETRY_STATUS = ("읽기실패", "오류")


def resume_manifest(path: Path) -> set[str]:
    """이어 하기용 기록(crops.csv·targets.csv)에서 끝난 key를 읽는다.

    작업을 중간에 끊으면 마지막 줄이 반쯤만 적혀 있을 수 있다 → 그 줄은 잘라 내고(다시 하게 됨),
    끝 칸(seconds)이 빈 줄도 끝난 것으로 치지 않는다.
    '읽기실패'·'오류'(메모리 부족 같은 일시적 실패)는 다시 한다. 다시 한 결과는 아래에 새 줄로 붙고,
    기록을 읽는 쪽은 status == 'ok' 줄만 쓰므로 옛 실패 줄은 그대로 둬도 된다.
    """
    if not path.exists():
        return set()
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        with open(path, "r+b") as fh:
            fh.truncate(data.rfind(b"\n") + 1)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {r["key"] for r in csv.DictReader(fh)
                if r.get("key") and r.get("seconds") and not r.get("status", "").startswith(RETRY_STATUS)}


def summary(pairs: list[Pair]) -> str:
    from collections import Counter
    st = Counter(p.status for p in pairs)
    ok = [p for p in pairs if p.status == "ok"]
    sp = Counter(p.split for p in ok)
    yr = Counter(p.date[:4] for p in ok)
    lines = [f"폴더 {len({p.folder for p in pairs})}개 · 원본 {len(pairs)}장 · 상태 {dict(st)}",
             f"쓸 수 있는 짝 {len(ok)}쌍 · 학습 {sp['train']} / 검증 {sp['val']}",
             f"안경 {sum(p.glasses for p in ok)} · PSD {sum(p.psd for p in ok)} · 여러 짝 폴더 {sum(p.multi for p in ok)}"
             f" · 표시 {dict(Counter(p.tag for p in ok if p.tag))}",
             "연도별 " + ", ".join(f"{k}:{v}" for k, v in sorted(yr.items()))]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    work = check_output(a.work, (a.src,))
    pairs = build_index(a.src, progress=lambda i, n: print(f"  {i}/{n}", flush=True))
    out = work / "index.csv"
    write_index(pairs, out)
    print(summary(pairs))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

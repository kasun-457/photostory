"""models.yaml에 적힌 모델 파일을 내려받고 sha256을 확인한다.

사용: python tools/fetch_models.py [--dest 폴더]
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "photostory" / "models" / "models.yaml"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=ROOT / "photostory" / "models" / "files")
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    roles = yaml.safe_load(CFG.read_text("utf-8"))["roles"]
    bad = 0
    for role, c in roles.items():
        if not c.get("url"):
            continue
        p = a.dest / c["file"]
        if p.exists() and sha256(p) == c["sha256"]:
            print(f"있음  {role}: {p.name}")
            continue
        print(f"받는 중 {role}: {c['url']}")
        tmp = p.with_suffix(p.suffix + ".part")
        urllib.request.urlretrieve(c["url"], tmp)
        got = sha256(tmp)
        if got != c["sha256"]:
            print(f"  sha256 불일치: {got}", file=sys.stderr)
            tmp.unlink()
            bad += 1
            continue
        tmp.replace(p)
        print(f"  완료 {p.stat().st_size // 1024} KB")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

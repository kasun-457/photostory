"""3-1 짝 목록과 원본 폴더 보호 (학습설계 0-1, 2)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training import index_pairs as ix
from training.safety import UnsafeOutput, check_output, is_inside


def test_output_inside_protected_drive_is_refused():
    with pytest.raises(UnsafeOutput):
        check_output(r"D:\photostory_data")
    with pytest.raises(UnsafeOutput):
        check_output(r"d:\1_촬영\3_증명사진\x")


def test_output_inside_source_is_refused(tmp_path):
    src = tmp_path / "src"
    with pytest.raises(UnsafeOutput):
        check_output(src / "work", (src,))
    assert check_output(tmp_path / "work", (src,)) == tmp_path / "work"


def test_is_inside_does_not_match_prefix_names(tmp_path):
    assert not is_inside(tmp_path / "src2", tmp_path / "src")
    assert is_inside(tmp_path / "src" / "a", tmp_path / "src")


def _jpg(path: Path, w=40, h=60):
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(path)


def _tree(src: Path):
    def cust(rel, files):
        d = src / rel
        d.mkdir(parents=True)
        for f, size in files.items():
            (_jpg(d / f, *size) if f.endswith(".jpg") or f.endswith(".JPG") else (d / f).write_bytes(b"x"))
    cust("2025/01/20250105_홍길동_증명", {"홍길동_원본.jpg": (40, 60), "홍길동_수정본.jpg": (40, 60),
                                       "홍길동_전송.jpg": (10, 15), "전송_홍길동_원본.jpg": (10, 15)})
    cust("2025/06/20250610_김철수_증명", {"김철수_원본.JPG": (40, 60), "김철수_수정본.jpg": (40, 60),
                                       "김철수_안경.jpg": (40, 60)})
    cust("2025/06/20250611_이영희_증명", {"이영희_원본.jpg": (40, 60)})
    # 한 폴더에 두 명, 끝에 _x 표시
    cust("2025/02/20250201_최가은_최나은_주민증_반명함_x",
         {"최가은_원본.jpg": (40, 60), "최가은_수정본.jpg": (40, 60), "최가은_수정과정.psd": (1, 1),
          "최나은_원본.jpg": (40, 60), "최나은_수정본.jpg": (40, 60)})
    cust("2025/06/20250612_박민수_증명", {"박민수_원본.jpg": (40, 60), "박민수_수정본.jpg": (41, 60)})
    # 같은 이름이 예전에도 왔음 → 둘 다 검증 쪽
    cust("2024/03/20240301_김철수_증명", {"김철수_원본.jpg": (40, 60), "김철수_수정.jpg": (40, 60)})
    (src / "2025" / "06" / "메모.txt").write_text("x")


def test_build_index_pairs_and_split(tmp_path):
    src = tmp_path / "src"
    _tree(src)
    before = sorted((p, p.stat().st_mtime_ns, p.stat().st_size) for p in src.rglob("*"))
    pairs = {(p.folder.replace("\\", "/"), p.name): p for p in ix.build_index(src, workers=2)}
    one = {k[0]: v for k, v in pairs.items()}
    after = sorted((p, p.stat().st_mtime_ns, p.stat().st_size) for p in src.rglob("*"))
    assert before == after                      # 원본 폴더는 그대로

    hong = one["2025/01/20250105_홍길동_증명"]
    assert (hong.original, hong.retouched, hong.status) == ("홍길동_원본.jpg", "홍길동_수정본.jpg", "ok")
    assert (hong.width, hong.height, hong.split) == (40, 60, "train")

    kim = one["2025/06/20250610_김철수_증명"]
    assert kim.original == "김철수_원본.JPG" and kim.glasses and kim.split == "val"
    old_kim = one["2024/03/20240301_김철수_증명"]
    assert old_kim.retouched == "김철수_수정.jpg" and old_kim.split == "val"

    assert one["2025/06/20250611_이영희_증명"].status == "수정본없음"
    assert one["2025/06/20250612_박민수_증명"].status == "크기다름"

    two = "2025/02/20250201_최가은_최나은_주민증_반명함_x"
    a, b = pairs[(two, "최가은")], pairs[(two, "최나은")]
    assert a.status == b.status == "ok" and a.multi and b.multi
    assert a.tag == "x" and a.psd and not b.psd
    assert not hong.multi and hong.tag == ""


def test_index_round_trip(tmp_path):
    src = tmp_path / "src"
    _tree(src)
    pairs = ix.build_index(src, workers=1)
    ix.write_index(pairs, tmp_path / "w" / "index.csv")
    back = ix.read_index(tmp_path / "w" / "index.csv")
    assert back == pairs


def test_resume_drops_half_written_last_line(tmp_path):
    from training.index_pairs import resume_manifest
    m = tmp_path / "targets.csv"
    m.write_bytes("key,status,seconds\r\naaa,ok,1.5\r\nbbb,ok,\r\nccc,o".encode("utf-8-sig"))
    assert resume_manifest(m) == {"aaa"}                 # 끝 칸이 빈 줄·반쯤 적힌 줄은 다시 함
    assert m.read_bytes().endswith(b"bbb,ok,\r\n")       # 반쯤 적힌 줄은 잘라 냄 → 다음 줄이 붙지 않음
    assert resume_manifest(tmp_path / "none.csv") == set()
    rows = ["key,status,seconds", "aaa,ok,1.5", "ddd,읽기실패:error,0.1", "eee,오류:MemoryError,0.2",
            "eee,ok,1.4", "fff,얼굴못찾음,0.3", ""]
    m.write_bytes("\r\n".join(rows).encode("utf-8-sig"))
    assert resume_manifest(m) == {"aaa", "eee", "fff"}  # 일시적 실패만 다시 (다시 해서 ok가 된 것은 끝남)

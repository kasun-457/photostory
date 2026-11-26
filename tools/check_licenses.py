"""판매용 빌드 전 검사: 상업 사용이 확인되지 않은 모델이 있으면 실패한다 (설계서 6-2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photostory.models.hub import ModelHub  # noqa: E402

bad = ModelHub().non_commercial()
if bad:
    print("상업 사용이 확인되지 않은 모델:", ", ".join(bad))
    sys.exit(1)
print("모든 모델이 상업 사용 가능으로 표시돼 있습니다")

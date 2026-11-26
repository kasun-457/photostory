# 사진이야기 증명사진 자동 보정

손님 정보 입력 → 사진 업로드 → 자동 보정 → 결과 확인 → 출력(`{이름}_수정본.jpg`, `{이름}_전송.jpg`, `{이름}_a4출력.jpg`).

- 설계: [docs/설계서.md](docs/설계서.md)
- 진행 상황: [docs/진행상황.md](docs/진행상황.md)

## 실행

```bash
pip install -r requirements.txt
python tools/fetch_models.py      # 모델 파일 내려받기 (처음 한 번)
python -m photostory
```

Linux에서는 MediaPipe용 `libegl1`이 필요합니다 (`apt-get install libegl1`).

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python tools\fetch_models.py
.venv\Scripts\python -m photostory
```

## 테스트

```bash
pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

## 구조

```
photostory/
  app/        화면 (PySide6). 표시만 하고 계산은 core에 맡김
  core/       파이프라인 엔진, 세션, 규격, 읽기·쓰기, 측정, 출력
  steps/      보정 단계 하나 = 파일 하나 (설계서 3-2)
  models/     역할 틀, 모델 허브, 어댑터, models.yaml
  specs/      기본 규격 14종 (JSON)
tools/        모델 내려받기, 라이선스 검사
tests/
```

고객 사진과 모델 파일은 git에 올리지 않습니다.

# VisionGuide 실행 순서 (팀원용)

---

## 처음 세팅하는 경우

### 1단계 — 레포 클론

```powershell
git clone https://github.com/milk152milk/navigation.git
```

---

### 2단계 — Python 환경 세팅 (최초 1회)

Python 3.10 이상이 설치되어 있어야 합니다.

```powershell
conda create -n visionguide python=3.10
conda activate visionguide
pip install -r requirements.txt
```

> GPU가 있는 경우 아래 명령으로 PyTorch를 CUDA 버전으로 교체하면 추론 속도가 10배 이상 빨라집니다:
> ```powershell
> pip uninstall torch torchvision -y
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

### 3단계 — 비공개 파일 받기

아래 파일들은 Git에 올라가 있지 않습니다. 팀 공유 드라이브(또는 카톡/슬랙)에서 받아서 해당 위치에 놓습니다.

| 파일 | 위치 |
|---|---|
| `.env` | `visionguide/` 루트 |
| `ngrok.exe` | `visionguide/tools/` |
| `best.pt` (장애물 모델) | `visionguide/ai_models/yolo_obstacle/weights/` |
| `best.pt` (노면 모델) | `visionguide/ai_models/yolo_surface/weights/` |
| `local.properties` (Android) | `visionguide/android/SafeStep/` |

`.env` 내용 예시:
```
NGROK_AUTH_TOKEN=토큰값
NGROK_DOMAIN=고정도메인.ngrok-free.app
TMAP_API_KEY=티맵API키
OLLAMA_MODEL=llama3.2
```

`local.properties` 내용 예시:
```
TMAP_API_KEY=티맵API키
```

---

### 4단계 — 서버 실행

프로젝트 **루트(visionguide/)** 에서 실행합니다.

```powershell
conda activate visionguide
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

아래와 같이 출력되면 정상입니다:

```
==================================================
🦮 VisionGuide 서버 시작
==================================================
📱 핸드폰 접속: https://고정도메인.ngrok-free.app
💻 노트북 대시보드: http://localhost:8000/dashboard
==================================================
```

> ngrok은 서버 시작 시 **자동으로 실행**됩니다. 별도로 실행할 필요 없습니다.

---

### 5단계 — Android 앱 빌드 (최초 1회)

1. Android Studio에서 아래 폴더를 엽니다:
   ```
   visionguide/android/SafeStep
   ```
2. `local.properties` 파일이 해당 폴더에 있는지 확인
3. 폰을 USB로 연결한 뒤 상단 ▶ Run 버튼 클릭 (또는 **Shift+F10**)
4. 앱 최초 실행 시 아래 권한을 모두 **허용** 해주세요:
   - 카메라
   - 마이크
   - 위치

---

### 6단계 — 앱에 서버 URL 입력

앱을 처음 실행하면 서버 URL 입력 화면이 자동으로 뜹니다.

4단계 서버 로그에서 확인한 ngrok 주소를 입력합니다:

```
https://고정도메인.ngrok-free.app
```

**연결** 버튼을 누르면 저장되며, 이후 실행 시에는 자동으로 연결됩니다.

---

### 7단계 — 연결 확인

브라우저에서 아래 주소 접속:

```
https://고정도메인.ngrok-free.app/health
```

아래와 같이 뜨면 정상 연결된 것입니다:

```json
{"status": "ok"}
```

---

## 대시보드 사용 (라이브 카메라 탭)

`http://localhost:8000/dashboard`

- 핸드폰 2대까지 자동으로 **서버1 / 서버2** 슬롯에 할당됨
- 화면 우측 상단 `서버1` `서버2` 버튼으로 활성 슬롯 전환
  - 점이 녹색 = 핸드폰 연결됨 / 회색 = 비어있음
- 헤더의 `Depth on/off` `데이터 수집 on/off` `Ollama on/off` `음성 on/off` 토글 활용
- **데이터 수집**을 ON으로 두면 5초당 1장씩 학습용 원본 + 라벨 JSON + 노면 마스크가 `data/captures/<날짜>/`에 저장됨 (디스크 5GB 한도 자동 관리)

> 동시 접속은 최대 2명. 3번째 핸드폰이 연결을 시도하면 앱에서 "서버가 만석입니다" 음성이 나오고, 30초 동안 핸드폰 1·2 중 하나가 요청을 안 보내면 슬롯이 비워져 자동 진입됩니다.

---

## 매일 실행할 때

```
① conda activate visionguide
② visionguide/ 루트에서 uvicorn backend.main:app --host 0.0.0.0 --port 8000 실행
③ 서버 로그에서 ngrok URL 확인
④ 앱 실행 (URL이 고정 도메인이면 재입력 불필요)
⑤ 대시보드: http://localhost:8000/dashboard
```

> ngrok 고정 도메인을 사용하면 서버를 재시작해도 URL이 바뀌지 않습니다.
> URL이 바뀐 경우에만 앱 시작 시 입력 다이얼로그가 자동으로 다시 뜹니다.

---

## 서버 강제 종료

```powershell
Get-Process python | Stop-Process -Force
```

---

## 방화벽 설정 (연결 안 될 때 1회 실행, 관리자 PowerShell)

```powershell
New-NetFirewallRule -DisplayName "VisionGuide Server" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

---

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| `conda` 명령어를 못 찾음 | Anaconda Prompt 또는 PowerShell에서 conda 초기화 필요 |
| `ModuleNotFoundError: No module named 'backend'` | `visionguide/` **루트**에서 uvicorn 실행 필요 |
| ngrok 실행 실패 | `tools/ngrok.exe` 위치 확인, `.env` 토큰 확인 |
| 앱에서 `/detect` 연결 실패 | 서버 실행 중인지 확인, 앱 URL 설정 확인 |
| AI 추론이 안 됨 | `ai_models/` 안에 `.pt` 모델 파일이 있는지 확인 |
| 분석이 매우 느림 + "거리 측정 일시 중단" 음성 | CPU 부하로 서버가 Depth를 자동 OFF한 상태. GPU PC에서 재실행 권장 (60초 후 자동 재시도됨) |
| 앱에서 "서버가 만석입니다" 음성 | 다른 핸드폰 2대 연결 중. 30초 미요청이면 자동 해제됨 |
| 대시보드 데이터 수집이 "디스크꽉참"으로 표시 | 5GB 한도 도달 + 정리 실패. `data/captures/` 수동 삭제 후 다시 ON |
| Android 빌드 오류 (`TMAP_API_KEY`) | `android/SafeStep/local.properties` 파일 확인 |

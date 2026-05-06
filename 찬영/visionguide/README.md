# 🦮 VisionGuide

> 시각장애인을 위한 실시간 AI 보행 비서
> 스마트폰 카메라 → AI 추론(장애물 탐지 + 노면 분류 + 거리 추정) → 음성·진동 안내

---

## ✅ 현재 구현된 기능

### 📱 SafeStep Android 앱

**탐지·안내**
- 실시간 카메라 분석 (서버 `/detect` + `/segment` 호출)
- 다객체 탐지 통합: 위험도 점수(면적 × 중심 가까움 × 신뢰도 × 그룹 가중치) 1위만 음성, 나머지는 진동
  - 같은 그룹 ≥ 3개 → "여러 명/여러 대"로 일반화
  - 1m 이하 ≥ 2개 → 가장 가까운 1개만 긴급 음성, 나머지 강진동
  - 5m 진입 ≥ 2개 → "여러 장애물 접근. 주의하세요"
- 거리 단계별 알림 (5m 음성+진동 약 / 3m 진동 중 / 1m 긴급 음성+진동 강)
- 진동 코얼레싱 (1초 슬롯 내 약한 진동 무시, 강 등급은 약·중을 끊고 즉시)
- 노면 진입 음성 (인도/차도/골목/횡단보도/주의구역 21종 세부)
- 차도·자전거도로 5초마다 방향 유도 음성 (왼쪽/오른쪽 노면 분석 기반)
- 횡단보도 신호등 색상 안내 (초록/빨강/인식불가)
- 차량 부재 시 "건너셔도 됩니다" 추가 안내

**상태 모니터링·자동 복구**
- 서버 연결/끊김/만석(429) 감지 + 30초 재연결 알림
- GPS on/off + 정확도 50m 초과 30초 → "실내인 것 같습니다"
- Depth 상태: 클라이언트 자체 추적 + 서버측 자동 OFF/ON 안내
- 카메라 어두움 감지(평균 밝기 5초) → 손전등 자동 ON, 밝아지면 자동 OFF
- 카메라 가림 감지 (5초 연속 빈 결과) → 안내
- 배터리 30/15/5% 단계별 안내 + 15% 이하 자동 절전 모드
  - 절전 모드: `/segment` 비활성, 5초→10초 반복, 화면 꺼짐 허용, 나침반 센서 주기 다운
- 앱 종료/백그라운드 시 손전등 강제 OFF

**네비게이션**
- T-map 보행자 길찾기 + 음성 인식 목적지 입력
- 도보 2km 또는 30분 초과 시 "가까운 정류장으로 안내할까요?" 옵션 제시
  - "네" → 가장 가까운 버스정류장 (없으면 지하철역) 자동 검색·재경로
  - 10초 무응답 → 도보 안내 진행
- 반대 방향 진행 감지 ("반대 방향입니다. 돌아서세요")
- 경로 이탈 자동 재탐색
- 동영상 파일 테스트 모드 (`VideoTestActivity`)

### 🤖 AI 추론 (서버)

- YOLO 장애물 탐지 (차량 / 자전거 / 사람 / 고정 장애물 / 신호등 등)
- YOLO 노면 세그멘테이션 (인도 / 차도 / 골목 / 점자블록 등 21종)
- Depth-Anything-V2 거리 추정 (미터 단위)
- **서버 부하 기반 Depth 자동 OFF/ON** (5회 평균 추론 시간 > 250ms → OFF, 60초 후 재시도, < 150ms → ON)
- 장애물 방향 감지 (왼쪽 / 정면 / 오른쪽)
- 노면 3구역 분석 (왼쪽 / 정면 / 오른쪽 각각 노면 판별)
- 신호등 색상 HSV 분석 (red / green)
- Ollama LLaMA 자연어 안내 (대시보드에서 on/off)
- 동일 프레임 중복 추론 방지 (MD5 캐시 + asyncio.Event)
- **동시 접속 최대 2명 제한** (X-Client-Id 기반 슬롯, 30초 미요청 시 자동 해제, 초과 시 429)

### 💻 노트북 대시보드

- 실시간 영상 + 분석 결과 표시
- **서버1 / 서버2 슬롯 토글**: 두 핸드폰 영상 별도 MJPEG로 분리 송출, 버튼으로 전환
  - 슬롯 점등(녹색=연결됨, 회색=비어있음) 2초마다 자동 폴링
- 영상 / 사진 파일 업로드 분석 (시크바, 노면 마스크 오버레이)
- AI 분석 카드: TTS 메시지 + 판단 근거 (장애물 / 노면 3구역 / 회피 방향)
- **데이터 수집 토글** (Depth 옆): 5초당 1장씩 원본 JPEG + 메타 JSON + 노면 마스크 PNG 저장 (학습 데이터 수집용)
  - 디스크 한도 5GB, 초과 시 가장 오래된 날짜 폴더 자동 삭제
- TTS / Ollama / Depth on/off 토글
- ngrok 자동 실행 (서버 시작 시 고정 URL 자동 발급)

---

## 📦 1. 설치

### Python 환경 세팅

```bash
# 1) 가상환경 생성 (conda 권장)
conda create -n visionguide python=3.10
conda activate visionguide

# 2) 패키지 설치
pip install -r requirements.txt
```

> ⚠️ Python **3.10 이상** 필요

### PyTorch (GPU 사용 시 속도 대폭 향상)

GPU가 있는 경우 CUDA 버전으로 재설치하면 추론 속도가 10배 이상 빨라집니다:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> GPU 없어도 실행은 됩니다. CPU에서는 Depth 추정이 프레임당 3~8초로 느려서 서버가 자동으로 Depth를 OFF시킬 수 있습니다.

### Ollama 설치 (선택 - LLaMA 자연어 안내)

1. https://ollama.com/download 에서 Windows용 설치
2. 설치 후 모델 다운로드 (1회만):

```bash
ollama pull llama3.2
```

> Ollama가 없어도 서버는 동작합니다. 안내 메시지가 rule-based로 대체됩니다.
> 대시보드에서 **Ollama on/off** 버튼으로 실시간 전환 가능합니다.

---

## ⚙️ 2. 환경 설정

### 서버 환경변수 (.env)

프로젝트 루트에 `.env` 파일을 팀장에게 받아 위치시킵니다:

```
visionguide/
└── .env   ← 여기
```

`.env` 내용 형식:

```
NGROK_AUTH_TOKEN=토큰값
NGROK_DOMAIN=고정도메인.ngrok-free.app
TMAP_API_KEY=티맵API키
OLLAMA_MODEL=llama3.2
```

> `.env`는 `.gitignore`에 포함되어 Git에 올라가지 않습니다.

### Android 앱 API 키 (local.properties)

SafeStep Android 앱 빌드 시 T-map API 키가 필요합니다.
`android/SafeStep/local.properties` 파일을 팀장에게 받아 위치시킵니다:

```
android/SafeStep/
└── local.properties   ← 여기
```

`local.properties` 내용 형식:

```
TMAP_API_KEY=티맵API키
```

> 빌드 시 `BuildConfig.TMAP_API_KEY`로 자동 주입됩니다.

---

## 🔧 3. ngrok 실행 파일 배치

`tools/ngrok.exe` 파일을 팀장에게 받아 아래 위치에 놓습니다:

```
visionguide/
└── tools/
    └── ngrok.exe
```

> ngrok은 서버 시작 시 **자동으로 실행**됩니다. 별도로 실행할 필요 없습니다.

---

## 🤖 4. AI 모델 배치

YOLO 모델 파일(`.pt`)을 아래 경로에 배치합니다:

```
visionguide/
└── ai_models/
    ├── yolo_obstacle/weights/best.pt   ← 장애물 탐지 모델
    └── yolo_surface/weights/best.pt    ← 노면 세그멘테이션 모델
```

> Depth-Anything-V2 모델은 첫 실행 시 HuggingFace에서 자동 다운로드됩니다 (인터넷 필요, 약 100MB).

---

## 🚀 5. 서버 실행

프로젝트 **루트(visionguide/)** 에서:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

성공하면 콘솔에 다음과 같이 나타납니다:

```
==================================================
🦮 VisionGuide 서버 시작
==================================================
📱 핸드폰 접속: https://고정도메인.ngrok-free.app
💻 노트북 대시보드: http://localhost:8000/dashboard
==================================================
```

---

## 📱 6. SafeStep Android 앱 사용법

1. Android Studio에서 `android/SafeStep/` 폴더 열기
2. `local.properties`에 T-map API 키 추가 (위 환경 설정 참고)
3. 앱 빌드 후 실행 (Android 8.0+ 권장)
4. 권한 허용: 카메라 / 마이크 / 위치
5. 앱 최초 실행 시 **ngrok URL 입력 다이얼로그** 자동 표시
   - 서버 로그의 ngrok URL 입력 (예: `https://abc123.ngrok-free.app`)
   - 입력값이 SharedPreferences에 저장됨 → 다음부터는 자동 연결
6. **카메라 모드**: 실시간 장애물 탐지 + 음성·진동 안내
7. **목적지 음성 입력**: 마이크 버튼 → 음성으로 목적지 → T-map 도보 경로 안내
8. **동영상 테스트**: `영상 테스트` 버튼으로 저장된 영상 파일 검증

> 서버 URL이 바뀌면 앱 시작 시 자동으로 입력 다이얼로그가 다시 뜹니다.

---

## 💻 7. 대시보드 사용법

`http://localhost:8000/dashboard` 접속 (노트북에서)

### 라이브 카메라 탭

- 두 핸드폰이 각각 슬롯 1, 2에 자동 할당됨
- 헤더 우측 **`서버1` / `서버2`** 버튼으로 활성 슬롯 전환
  - 점 색상: 녹색 = 핸드폰 연결됨 / 회색 = 비어있음
- 활성 슬롯의 카메라 + 노면 마스크 + 장애물 박스 + AI 분석 카드 표시

### 영상 / 사진 분석 탭

- 영상 / 사진 파일 업로드 → AI 분석
- 시크바로 프레임별 탐색
- 노면 마스크 오버레이 + 장애물 박스 표시 (`/segment?include_mask=1`)

### 헤더 토글

| 버튼 | 기능 |
|---|---|
| **Depth on/off** | Depth-Anything 거리 추정 사용 여부 |
| **데이터 수집 on/off** | 5초/장씩 학습용 원본 + 메타·마스크 저장 (기본 OFF) |
| **Ollama on/off** | LLaMA 자연어 안내 여부 (off 시 rule-based) |
| **음성 on/off** | 대시보드 TTS 출력 |
| **대시보드 연결됨** | WebSocket 상태 표시 |

### 사이드바 카드

- **AI 분석**: 현재 슬롯의 TTS 메시지 + 판단 근거
- **노면 분석**: 21개 클래스 체크박스 (필터)
- **장애물 분석**: 그룹별 체크박스 (필터)
- **데이터 수집**: 저장 상태 / 누적 장수 / 용량 / 진행바 / 폴더 경로

---

## 📂 8. 학습 데이터 수집

대시보드 헤더의 **`데이터 수집` 핀**을 누르면 ON됩니다 (기본 OFF).

### 저장 형식

```
data/captures/
└── 2026-05-04/
    ├── 08-22-36_slot1_a3f2b1c4_000123.jpg     ← 원본 (재인코딩 X)
    ├── 08-22-36_slot1_a3f2b1c4_000123.json    ← 의사 라벨 메타데이터
    └── 08-22-36_slot1_a3f2b1c4_000123_mask.png ← 노면 학습용 단일채널 마스크
```

### JSON 메타데이터

```json
{
  "ts": "2026-05-04T08:22:36.697",
  "client_id": "uuid",
  "slot": 1,
  "md5": "a3f2b1c4",
  "img_width": 640, "img_height": 480,
  "obstacles": [
    {"cls_name": "car", "group": "vehicle", "conf": 0.87,
     "bbox": [100, 200, 300, 400], "depth_m": 4.5,
     "direction": "왼쪽", "light_color": null}
  ],
  "surface_zones": {"정면": {"cls_name": "sidewalk_blocks", "group": "safe"}},
  "surface_class_ratios": {"sidewalk_blocks": 65.3},
  "depth_status": {"enabled": true, "auto_off": false}
}
```

### 정책

- **저장 간격**: 슬롯별 5초당 1장 (두 핸드폰 독립 카운터)
- **필터 없음**: 탐지 결과가 비어도 저장 (false negative도 학습에 필요)
- **디스크 한도**: 5GB (초과 시 가장 오래된 날짜 폴더부터 자동 삭제, 정리 후에도 초과면 자동 OFF)
- **동시성 안전**: `_frame_idx`는 lock으로 보호, 디스크 I/O는 `asyncio.to_thread()`로 백그라운드 실행 (이벤트 루프 블록 X)

> ⚠️ 라벨은 **현재 모델의 추론 결과(의사 라벨)** 이므로 학습 시 사람 검수·필터링 권장.

---

## 🗂️ 9. 폴더 구조

```
visionguide/
├── android/
│   └── SafeStep/
│       └── app/src/main/
│           ├── AndroidManifest.xml         # SplashActivity / MapActivity / VideoTestActivity
│           ├── java/com/safestep/app/
│           │   ├── SplashActivity.kt       # ngrok URL 입력, 모드 선택
│           │   ├── MapActivity.kt          # 메인: 카메라 + 지도 + 내비
│           │   ├── VideoTestActivity.kt    # 동영상 파일 테스트
│           │   ├── BoundingBoxOverlay.kt
│           │   ├── detect/
│           │   │   ├── ObjectDetector.kt
│           │   │   ├── RemoteDetector.kt   # /detect 호출, X-Client-Id 헤더
│           │   │   ├── SegmentationClient.kt # /segment 호출
│           │   │   └── Detection.kt
│           │   └── navigation/
│           │       ├── TmapService.kt      # POI / 도보 경로 / 정류장 검색
│           │       └── NavigationGuide.kt  # 턴바이턴 + 반대방향 감지
│           └── res/layout/
│               ├── activity_splash.xml
│               ├── activity_map.xml
│               └── activity_video_test.xml
│
├── backend/
│   ├── main.py                             # FastAPI + ngrok 자동 실행
│   ├── routers/
│   │   ├── camera_router.py                # WebSocket + 슬롯별 MJPEG 스트림
│   │   ├── safestep_router.py              # /detect, /segment, /capture/*
│   │   ├── video_router.py                 # 영상/사진 분석
│   │   └── nav_router.py                   # T-map 길찾기 (브라우저용)
│   ├── services/
│   │   ├── inference_service.py            # YOLO + Depth + Depth 자동 OFF/ON
│   │   └── capture_service.py              # 학습 데이터 캡처 (5초/장, 5GB 한도)
│   └── utils/
│       ├── image_utils.py
│       └── logger.py
│
├── frontend/
│   ├── index.html                          # 스마트폰용 (브라우저 모드)
│   ├── dashboard.html                      # 노트북 대시보드 (슬롯 토글, 데이터 수집)
│   ├── css/style.css
│   └── js/
│
├── ai_models/
│   ├── yolo_obstacle/weights/best.pt
│   └── yolo_surface/weights/best.pt
│
├── data/
│   └── captures/                           # 학습 데이터 (Git 제외)
│
├── tools/ngrok.exe                          # Git 제외
├── .env                                     # Git 제외
├── 상황별 반응.md                            # 음성·진동·시나리오 명세
├── VisionGuide_실행순서.md                   # 팀원용 빠른 실행 가이드
├── requirements.txt
└── README.md
```

---

## 🔌 10. API 엔드포인트

### HTTP

| 메서드 | URL | 설명 |
|---|---|---|
| GET | `/` | 스마트폰 브라우저용 메인 |
| GET | `/dashboard` | 노트북 대시보드 |
| GET | `/health` | 서버 상태 확인 |
| POST | `/detect` | **SafeStep** 장애물 탐지 (X-Client-Id 헤더 필수) |
| POST | `/segment` | **SafeStep** 노면 세그멘테이션. `?include_mask=1` 옵션 시 합쳐진 마스크 PNG 포함 |
| GET | `/camera/stream/{slot}` | 슬롯별 MJPEG 스트림 (대시보드 `<img>`용, slot=1 또는 2) |
| GET | `/camera/stream` | 레거시 — 슬롯 1로 fallback |
| GET | `/camera/slots` | 슬롯 활성 상태 조회 (대시보드 폴링) |
| POST | `/capture/toggle` | 학습 데이터 수집 ON/OFF 토글 |
| GET | `/capture/status` | 캡처 상태 (저장 수, 용량, 한도) |
| POST | `/depth-toggle` | Depth 추정 수동 ON/OFF |
| POST | `/ollama-toggle` | Ollama LLM ON/OFF |
| POST | `/analyze-file` | 영상/사진 파일 업로드 → AI 분석 |
| GET | `/api/nav/config` | T-map API 키 (브라우저 길찾기용) |
| POST | `/api/nav/search` | T-map POI 검색 |
| POST | `/api/nav/directions` | T-map 보행자 경로 |

### 응답 코드

- `200`: 정상
- `429`: 동시 접속 한도(2명) 초과 — 클라이언트는 30초 후 재시도 권장

### `/detect` 응답 포맷

```json
{
  "detections": [
    {
      "label":      "bicycle",
      "label_ko":   "micro",
      "confidence": 0.87,
      "box":        [120, 80, 340, 300],
      "direction":  "정면",
      "depth_m":    4.8
    }
  ],
  "message":      "4.8m 정면에 자전거가 있습니다. 왼쪽으로 피하세요.",
  "dodge":        "왼쪽",
  "analysis":     { "obstacle": {...}, "surface": {...} },
  "detail":       {...},
  "depth_status": { "enabled": true, "auto_off": false, "manual_off": false, "reason": "" }
}
```

> `dodge`: 회피 방향 (왼쪽/정면/오른쪽) — 진동 방향에 사용
> `direction`: 장애물 위치 방향 (장애물이 어디 있는지) — 표시용
> `depth_status.auto_off=true`이면 서버가 부하로 Depth를 임시 비활성화한 상태

### `/segment` 응답 포맷

기본:
```json
{
  "status":              "sidewalk",
  "ratios":              { "road": 0.1, "sidewalk": 0.8, "crosswalk": 0.0, "alley": 0.1 },
  "front_cls":           "sidewalk_blocks",
  "left_status":         "sidewalk",
  "right_status":        "road",
  "traffic_light_color": ""
}
```

`?include_mask=1` 추가 시:
```json
{
  ...,
  "mask_b64": "<base64 PNG>"
}
```

> `traffic_light_color`: `"red"` / `"green"` / `""` (crosswalk일 때만 의미 있음)
> 같은 프레임으로 `/detect`와 `/segment`를 동시 요청해도 추론은 1번만 실행됩니다 (MD5 캐시).
> mask_b64는 대시보드 영상 분석 탭 전용 — 안드로이드 앱은 보내지 않음 (대역폭 절감).

### `/camera/slots` 응답

```json
{
  "slots": [
    { "slot": 1, "active": true,  "client_id": "uuid-A" },
    { "slot": 2, "active": false, "client_id": "" }
  ]
}
```

### `/capture/status` 응답

```json
{
  "enabled":        false,
  "auto_disabled":  false,
  "total_count":    234,
  "total_bytes":    12582912,
  "total_mb":       12.0,
  "limit_mb":       5120.0,
  "interval_sec":   5.0,
  "folder":         "C:\\Users\\.../data/captures"
}
```

### WebSocket

| URL | 설명 |
|---|---|
| `ws://localhost:8000/ws/camera` | 스마트폰 (브라우저 모드) → 서버 |
| `ws://localhost:8000/ws/dashboard` | 서버 → 대시보드 (`frame_overlay` 브로드캐스트) |

#### `frame_overlay` 메시지에 포함되는 주요 필드

```json
{
  "type": "frame_overlay",
  "slot": 1,                        ← 어느 슬롯의 프레임인지
  "surface_masks": { "sidewalk_asphalt": "data:image/png;base64,..." },
  "obstacle_items": [...],
  "img_width": 640, "img_height": 480,
  "message": "...",
  "detail": {...},
  "analysis": { "obstacle": {...}, "surface": {...} }
}
```

대시보드는 `slot` 필드로 각 슬롯의 overlay 캐시를 분리 관리합니다.

---

## 📨 11. 안드로이드 클라이언트 헤더

`/detect`, `/segment` 호출 시 항상 같이 보내는 헤더:

| 헤더 | 값 | 용도 |
|---|---|---|
| `X-Client-Id` | UUID (앱 시작 시 1회 생성) | 서버 슬롯 추적 (1 or 2 자동 할당) |

> 헤더 없으면 서버가 IP로 fallback. 두 핸드폰이 같은 NAT 뒤에 있을 수 있으니 헤더 권장.

---

## 🏷️ 12. 감지 클래스 목록

### 장애물 모델 (yolo_obstacle)

| 그룹 | 클래스 |
|---|---|
| 차량 (vehicle) | car, bus, truck, motorcycle |
| 개인이동장치 (micro) | bicycle, scooter |
| 사람/동물 (person) | person, cat, dog, stroller, wheelchair, carrier, movable_signage |
| 고정 장애물 (fixed) | barricade, bench, bollard, chair, fire_hydrant, kiosk, parking_meter, pole, potted_plant, power_controller, stop, table, tree_trunk, traffic_light_controller |
| 신호등/표지판 (signal) | traffic_light, traffic_sign |

### 노면 모델 (yolo_surface)

| 그룹 | 클래스 |
|---|---|
| safe (안전) | sidewalk_asphalt, sidewalk_blocks, sidewalk_cement, sidewalk_urethane, sidewalk_soil_stone, sidewalk_other |
| danger (차도) | roadway_normal, roadway_crosswalk |
| caution (주의) | alley_normal, alley_crosswalk, alley_damaged, alley_speed_bump, sidewalk_damaged, caution_zone_grating, caution_zone_manhole, caution_zone_repair_zone, caution_zone_stairs, caution_zone_tree_zone |
| braille (점자블록) | braille_guide_blocks_normal, braille_guide_blocks_damaged |
| bike (자전거도로) | bike_lane |

### 위험도 점수 그룹 가중치 (안드로이드 측)

| 그룹 | 가중치 |
|---|---|
| 차량 | 1.0 |
| 개인이동장치 | 0.9 |
| 사람/동물 | 0.7 |
| 고정장애물 | 0.5 |
| 신호등/표지판 | 0.3 |

`위험도 = 면적 × 중심 가까움 × 신뢰도 × 그룹 가중치` — 한 프레임에서 1위만 음성 발화.

---

## ⚙️ 13. 주요 상수

### 서버 (`backend/services/inference_service.py`)

```python
DEPTH_SCALE = 0.65                    # Depth 모델이 멀게 측정하는 경향 보정
_DEPTH_AUTO_OFF_MS  = 250.0           # 5회 평균 250ms 초과 → Depth 자동 OFF
_DEPTH_AUTO_ON_MS   = 150.0           # 평균 150ms 이하 → 자동 ON
_DEPTH_RECHECK_SEC  = 60.0            # OFF 후 60초 뒤 재시도

_APPROACH_SPEED_THRESHOLD = {
    "vehicle": 0.3, "micro": 1.2, "person": 2.0, "fixed": None, "signal": None,
}
_LLM_MIN_INTERVAL = 3.0               # Ollama 최소 호출 간격
```

### 서버 (`backend/routers/safestep_router.py`)

```python
SLOTS         = (1, 2)
MAX_CLIENTS   = 2                     # 동시 접속 한도
SESSION_TTL   = 30.0                  # 슬롯 자동 해제 시간
_CACHE_TTL    = 2.0                   # 같은 프레임 추론 결과 캐시
```

### 서버 (`backend/services/capture_service.py`)

```python
CAPTURE_ROOT      = Path("data/captures")
SAVE_INTERVAL_SEC = 5.0
DISK_LIMIT_BYTES  = 5 * 1024**3       # 5GB
PRUNE_TARGET_PCT  = 0.85
```

### 안드로이드 (`MapActivity.kt`)

```kotlin
WRONG_DIR_ANGLE_DEG    = 120.0   // 진행 방향과 경로 방향 각도 차이
WRONG_DIR_COOLDOWN_MS  = 10_000  // 반대 방향 음성 쿨다운
```

---

## 🛠️ 14. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| 스마트폰에서 카메라 권한이 안 뜸 | HTTPS가 아님. ngrok URL(https://...)로 접속 필수 |
| ngrok 실행 실패 | `tools/ngrok.exe` 위치, `.env` 토큰 확인 |
| WebSocket 계속 재연결 | 서버 실행 여부, 방화벽 8000 포트 확인 |
| `ModuleNotFoundError: No module named 'backend'` | 프로젝트 **루트에서** uvicorn 실행 |
| AI 추론이 안 됨 | `ai_models/` 안에 `.pt` 파일 확인 |
| 분석이 매우 느림 → "거리 측정 일시 중단" 음성 | CPU 환경 부하로 Depth 자동 OFF. GPU 권장 |
| **앱**: "서버가 만석입니다" 음성 | 다른 핸드폰 2대가 이미 연결 중. 30초 미요청이면 자동 해제 |
| **앱**: 빌드 오류 (`TMAP_API_KEY`) | `android/SafeStep/local.properties`에 키 추가 |
| **앱**: `/detect` 연결 실패 | Splash 화면에서 ngrok URL 재입력 |
| **앱**: 진동 없음 | Android 권한 확인 (`VIBRATE`) |
| **앱**: TTS 안 나옴 | 기기 볼륨 + 한국어 TTS 엔진 설치 (설정 → 접근성 → TTS) |
| **앱**: 손전등이 꺼지지 않음 | 앱 종료 또는 백그라운드 진입 시 자동 OFF — 앱 강제 종료 후에도 안 꺼지면 카메라 앱 재시작 |
| **대시보드**: 데이터 수집 라벨이 "디스크꽉참" | 5GB 한도 도달 + 자동 정리 실패. `data/captures/` 수동 삭제 후 다시 ON |
| **대시보드**: 영상 분석 탭에서 노면 마스크 안 보임 | `/segment?include_mask=1` 호출 확인 (브라우저 캐시 비우기) |
| **대시보드**: 슬롯 점이 회색인데 핸드폰은 연결돼 있다고 함 | 핸드폰의 `X-Client-Id`가 30초 동안 요청 안 보냈음 — 카메라 모드 재시작 |

---

## 📚 추가 문서

- **[상황별 반응.md](./상황별%20반응.md)** — 음성 우선순위, 진동 정책, 노면·장애물·네비·서버 상태별 반응 명세
- **[VisionGuide_실행순서.md](./VisionGuide_실행순서.md)** — 팀원용 빠른 실행 가이드
- **[android/SafeStep/app/src/main/assets/MODEL_SPEC.md](./android/SafeStep/app/src/main/assets/MODEL_SPEC.md)** — 온디바이스 PyTorch Mobile 모델 인터페이스 (현재는 RemoteDetector 사용 중이라 미사용)

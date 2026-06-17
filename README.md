# SafeStep 🦺

시각장애인을 위한 AI 보행 안전 앱

카메라로 주변 장애물·노면·신호등을 실시간 분석해 TTS와 진동으로 경고하며, Tmap 기반 도보 내비게이션을 제공합니다.

---

## 시스템 구성

```
[Android 앱] ──── HTTPS (ngrok) ────► [Python 서버 (PC)]
   카메라 프레임 전송                      YOLO 객체 탐지 (bbox.pt)
   TTS / 진동 경고                        노면 세그멘테이션 (surface.pt)
   Tmap 도보 내비게이션                    신호등 색상 감지 (HSV)
   osmdroid 지도                          거리·접근속도 추정 (Depth-Anything-V2)
```

서버는 PC에서 실행하며, ngrok을 통해 Wi-Fi 없이 어디서든 접속 가능합니다.

---

## 빠른 시작

자세한 실행 순서는 **[실행순서.md](./실행순서.md)** 를 참고하세요.

- **팀원**: 레포 클론 → SERVER_URL 입력 → 앱 빌드
- **서버 담당(준범)**: `python server.py` 실행 → URL 공유

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 객체 탐지 | 사람·자동차·자전거·볼라드 등 실시간 감지, 방향(왼쪽/정면/오른쪽) TTS |
| 거리 추정 | Depth-Anything-V2 기반 미터 단위 거리 측정 |
| 접근 속도 경고 | 차량·사람·자전거 접근 속도 계산, 정지 차량 오경보 방지 |
| 노면 세그멘테이션 | 보도·차도·횡단보도·골목·위험구역 색상 오버레이 |
| 계단 즉시 경고 | 계단 감지 시 최우선 TTS 발화 |
| 차도 방향 힌트 | 내비 중 차도 진입 시 "오른쪽 보도로 이동하세요" 안내 |
| 신호등 감지 | 빨강·초록·깜빡임 상태 변화 시 TTS |
| 보행 모드 | 앱 시작 즉시 장애물 경고 활성화 (목적지 없어도 동작) |
| 도보 내비게이션 | 음성으로 목적지 입력 → Tmap 턴바이턴 안내 |
| 잔여 경로선 | 지나온 경로 자동 삭제, 남은 구간만 표시 |
| 경로 이탈 감지 | 40m 이상 벗어나면 경고 + 자동 재탐색 |
| Heading-up 지도 | 나침반 기반 진행 방향이 항상 위 |
| 배터리 절약 | 가속도계 기반 정지 감지 → detection 주기 자동 축소 |
| ngrok 지원 | Wi-Fi 없이 LTE/5G 환경에서도 서버 접속 가능 |

---

## 노면 세그멘테이션 색상

| 색상 | 의미 | TTS 경고 |
|------|------|---------|
| 🟢 초록 | 보도 (안전) | 없음 |
| 🔴 빨강 | 차도 (위험) | "차도입니다. (방향) 보도로 이동하세요." |
| 🟡 노랑 | 횡단보도 / 골목 | "골목길입니다. 주의하세요." |
| 🟠 주황 | 위험구역 (계단·맨홀·격자) | "계단이 있습니다. 조심하세요." / "위험 구역입니다." |

---

## 프로젝트 구조

```
SafeStep/
├── server/
│   ├── server.py              # FastAPI 서버 (탐지·세그멘테이션·신호등·깊이 추정)
│   ├── .env                   # ngrok 토큰 (NGROK_AUTH_TOKEN) — Git 미포함
│   ├── bbox.pt                # YOLO 객체 탐지 모델
│   └── surface.pt             # 한국 도로 노면 세그멘테이션 모델
│
└── app/src/main/
    ├── java/com/safestep/app/
    │   ├── SplashActivity.kt          # 모드 선택 화면
    │   ├── MapActivity.kt             # 메인 (지도 + 카메라 + 내비)
    │   ├── VideoTestActivity.kt       # 동영상 테스트 모드
    │   ├── BoundingBoxOverlay.kt      # 바운딩박스 커스텀 뷰
    │   ├── detect/
    │   │   ├── RemoteDetector.kt      # ★ SERVER_URL 설정하는 곳
    │   │   ├── SegmentationClient.kt  # 노면 세그멘테이션 통신
    │   │   ├── SignalClient.kt        # 신호등 감지 통신
    │   │   ├── Detection.kt           # 탐지 결과 데이터 클래스
    │   │   └── ObjectDetector.kt
    │   └── navigation/
    │       ├── TmapService.kt         # Tmap POI·경로 API
    │       └── NavigationGuide.kt     # 턴바이턴 안내 로직
    └── res/layout/
        ├── activity_map.xml
        ├── activity_splash.xml
        └── activity_video_test.xml
```

---

## 서버 API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /detect` | YOLO 객체 탐지 + 거리·접근속도 반환 |
| `POST /segment` | 노면 세그멘테이션 + 3구역 분석 + 계단 감지 |
| `POST /signal` | 신호등 색상 감지 (HSV 분석) |
| `GET /health` | 서버 상태 확인 |

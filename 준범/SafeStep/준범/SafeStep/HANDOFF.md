# SafeStep — 세션 핸드오프 문서

> 다음 대화 시작할 때: **"`C:\Users\ParkJunbeom\AndroidStudioProjects\SafeStep\HANDOFF.md` 읽고 이어서 작업하자"** 라고만 말하면 됩니다. 폴더 마운트도 같이 요청하라고 알려주세요.

---

## 1. 프로젝트 개요

- **이름**: SafeStep
- **목적**: 시각장애인을 위한 AI 공간 통역 안드로이드 앱
- **동작**: 카메라로 주변 인식 → 가까이 오는 객체 감지 → 음성(TTS) + 진동 + 화면 배너로 안내
- **위치**: `C:\Users\ParkJunbeom\AndroidStudioProjects\SafeStep`
- **패키지**: `com.safestep.app`

## 2. 기술 스택

- **언어**: Kotlin
- **minSdk**: 26 / **targetSdk**: 36 / **compileSdk**: 36
- **카메라**: CameraX 1.3.1 (`Preview` + `ImageAnalysis`)
- **추론**: PyTorch Mobile **Lite Interpreter** 2.1.0 (`.ptl` 형식)
  - `org.pytorch:pytorch_android_lite:2.1.0`
  - `org.pytorch:pytorch_android_torchvision:2.1.0` (pytorch_android transitive 제외)
- **음성**: Android `TextToSpeech` (한국어) + `SpeechRecognizer`
- **레이아웃**: ConstraintLayout
- **테마 색상**: 배경 `#141414`, 강조 `#F97316` (오렌지), 경고 배너 동일 오렌지

## 3. 아키텍처 / 화면 흐름

```
SplashActivity (로고 + "시작하기" 버튼 + TTS 자동 안내)
        │
        ▼
MainActivity (카메라 프리뷰 + 위험 안내 + 마이크 버튼)
        │
        ├── ImageAnalysis use case → analyzeFrame()
        │       │
        │       ▼
        │   ObjectDetector.detect(bitmap, rotation)
        │       │
        │       ▼
        │   List<Detection> → currentDetections (음성명령에서 참조)
        │
        ├── handleDetections() → 가장 위험한 1개 선택
        │       │
        │       ▼
        │   showWarning("정면 매우 가까이 사람 접근", side)
        │       ├── 화면: warningBanner 표시 (3초)
        │       ├── 음성: TTS (한국어, 2.5초 쿨다운)
        │       └── 진동: 방향별 패턴 (왼/정면/오른)
        │
        ├── 더블탭 → repeatLastStatus() → 마지막 경고 또는 "현재 위험 없음"
        │
        └── 마이크 버튼 → SpeechRecognizer → "주변 알려줘" → speakCurrentDetections()
```

## 4. 현재 파일 구조

```
app/src/main/
├── AndroidManifest.xml              CAMERA·VIBRATE·RECORD_AUDIO 권한, SplashActivity launcher
├── java/com/safestep/app/
│   ├── SplashActivity.kt            TTS 자동 발화 ("SafeStep. 시작하기 버튼을 눌러주세요.")
│   ├── MainActivity.kt              전체 기능 통합 (아래 §5 참고)
│   └── detect/
│       ├── Detection.kt             data class (label, confidence, RectF box)
│       ├── ObjectDetector.kt        인터페이스 + NoOpDetector + PyTorchDetector
│       └── AssetUtils.kt            assets → cacheDir 복사 헬퍼
├── res/
│   ├── layout/
│   │   ├── activity_splash.xml      버튼 240dp×64dp, contentDescription 추가
│   │   └── activity_main.xml        하단 마이크 버튼 + 권한 거부 오버레이 추가
│   ├── drawable/                    btn_orange.xml, dot_orange.xml, dot_white.xml
│   └── values/                      colors.xml, strings.xml, themes.xml
└── assets/
    ├── MODEL_SPEC.md                팀원에게 줄 모델 인계 명세
    └── labels.txt                   임시 라벨 (팀 모델 받으면 교체)

app/build.gradle.kts                 pytorch_android_lite:2.1.0 사용
```

## 5. 구현된 기능 목록

### ✅ 세션 1 (이전)
1. SplashActivity 크래시 수정
2. ObjectDetector 인프라 (NoOpDetector / PyTorchDetector 골격)
3. CameraX ImageAnalysis → 추론 → TTS/진동 파이프라인
4. TTS 쿨다운 (2.5초)
5. PyTorch Lite 빌드 에러 수정

### ✅ 세션 2 (이번)
6. **SplashActivity TTS 자동 발화**: 앱 켜자마자 "SafeStep. 시작하기 버튼을 눌러주세요." 발화
7. **버튼 크기 확대**: 240dp×64dp, contentDescription 추가
8. **카메라 권한 거부 안내 화면**: 거부 시 permissionDeniedView 오버레이 표시 + "설정으로 이동" 버튼
9. **onResume 권한 재확인**: 설정에서 권한 허용 후 돌아오면 자동으로 카메라 시작
10. **더블탭 재발화**: GestureDetector 더블탭 → lastWarningMessage 또는 "현재 위험 없음" TTS
11. **방향별 진동 패턴**: 왼쪽(짧짧긴) / 정면(긴짧긴) / 오른쪽(긴짧짧) 차별화
12. **거리감 표현**: area ≥ 0.40 → "매우 가까이 사람 접근"
13. **음성 명령 (SpeechRecognizer)**: 마이크 FAB → "주변 알려줘" → 현재 감지 객체 TTS, "멈춰/중지" 명령도 지원
14. **RECORD_AUDIO 권한** AndroidManifest에 추가
15. **accessibilityLiveRegion="assertive"**: 경고 배너 TalkBack 연동

## 6. 다음 작업 (우선순위 순)

### A. 팀 모델 도착 시 (최우선)
- `app/src/main/assets/`에 `model.ptl` + `labels.txt` 추가
- `ObjectDetector.kt`의 `PyTorchDetector.parseRawOutput()` 구현
  - 옵션 A (raw YOLO [1,N,5+cls]): obj_conf × cls_score + NMS (NMS 헬퍼 이미 있음)
  - 옵션 B (boxes/scores/labels 튜플): 좌표 정규화만 하면 됨

### B. TalkBack 완성
- 모든 커스텀 뷰에 contentDescription 확인
- `AccessibilityNodeInfoCompat`으로 커스텀 액션 등록 (선택)

### C. 음성 명령 확장
- "도움말" → 지원 명령어 목록 TTS
- "감도 높여줘 / 낮춰줘" → DANGER_AREA 동적 조정
- "한국어 / 영어" → TTS 언어 전환

### D. UX 개선
- 감지 없는 시간 지속 시 "주변 안전합니다" 주기적 안내
- 배터리 절약 모드 (화면 꺼도 카메라+TTS 동작)

## 7. 알아둘 것 / 디버깅 팁

- **더블탭이 안 되면**: rootLayout `clickable=true` 확인. 하위 뷰가 터치를 소비하면 안 됨.
- **SpeechRecognizer 오류 5 (ERROR_CLIENT)**: 이미 startListening 중인데 다시 호출하면 발생. isListening 플래그로 방어 중.
- **음성 인식 미지원 기기**: `SpeechRecognizer.isRecognitionAvailable()` false면 micButton 자동 숨김.
- **`MODEL_SPEC.md`**: 팀원에게 그대로 공유. TorchScript Lite export 코드 예시 + 입력/출력 포맷.
- **로그캣 붙일 때**: `FATAL EXCEPTION` + `Caused by:` 두 줄만.
- **카메라 권한 거부 후 다시 허용**: 앱 정보 → 권한 → 카메라 허용 → 앱 포커스 돌아오면 onResume에서 자동 감지.
- **`getSystemService(VIBRATOR_SERVICE) as Vibrator`**: API 31 deprecated이지만 minSdk 26이라 유지. 추후 VibratorManager 마이그레이션 가능.

## 8. 다음 세션 시작 시 추천 프롬프트

```
C:\Users\ParkJunbeom\AndroidStudioProjects\SafeStep\HANDOFF.md 읽고 이어서 작업하자.
프로젝트 폴더는 C:\Users\ParkJunbeom\AndroidStudioProjects\SafeStep 다.
```

폴더 마운트 후, 사용자 응답에 따라 다음 단계 진행:
- "팀이 모델 줬어" → `parseRawOutput()` 구현
- "빌드 에러나" → 에러 메시지 받아서 수정
- "TalkBack 작업하자" → contentDescription / 커스텀 액션 추가

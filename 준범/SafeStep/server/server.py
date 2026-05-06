# SafeStep Detection Server
# - /detect  : YOLO11n 객체 탐지 (bbox2.pt)
# - /segment : YOLO 노면 세그멘테이션 (surface.pt) — 한국 도로 데이터 학습
# - /signal  : 신호등 색상 감지 (yolov8n COCO + HSV)
# - /health  : 서버 상태 확인
#
# ngrok 사용 시 어디서든 접속 가능 (.env 에 NGROK_AUTH_TOKEN / NGROK_DOMAIN 설정)

import base64
import hashlib
import io
import os

# OpenMP 중복 로드 방지 (conda 환경에서 libiomp5md.dll 충돌 억제)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

load_dotenv()   # server/.env 자동 로드

# GPU 지원 여부 자동 감지 (FP16은 아키텍처 호환성 문제로 비활성화)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_half   = False   # FP16 비활성화 — 일부 GPU 아키텍처에서 kernel 미지원
if _device == "cuda":
    print(f"[SafeStep] 🚀 GPU 감지: {torch.cuda.get_device_name(0)} — FP32 GPU 추론 활성화")
else:
    print("[SafeStep] CPU 모드로 실행 중 (GPU 없음)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO 객체 탐지 모델 (bbox2.pt)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_PATH = Path(__file__).parent / "bbox2.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"bbox2.pt 를 서버 폴더에 복사해주세요: {MODEL_PATH.parent}")

print(f"[SafeStep] YOLO 탐지 모델 로드 중: {MODEL_PATH}")
yolo_model = YOLO(str(MODEL_PATH))
print("[SafeStep] YOLO 탐지 로드 완료 ✅")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO 노면 세그멘테이션 모델 (surface.pt)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SURFACE_PATH = Path(__file__).parent / "surface.pt"
surface_model = None

if SURFACE_PATH.exists():
    print(f"[SafeStep] 노면 세그멘테이션 모델 로드 중: {SURFACE_PATH}")
    try:
        surface_model = YOLO(str(SURFACE_PATH))
        print("[SafeStep] 노면 세그멘테이션 로드 완료 ✅")
        print(f"[SafeStep] 클래스 목록: {surface_model.names}")
    except Exception as e:
        print(f"[SafeStep] 노면 모델 로드 실패: {e}")
else:
    print("[SafeStep] surface.pt 없음 — 서버 폴더에 복사해주세요")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 신호등 탐지 모델 (yolov8n — COCO, 자동 다운로드)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
signal_model = None
try:
    print("[SafeStep] 신호등 탐지 모델 로드 중 (yolov8n)...")
    signal_model = YOLO("yolov8n.pt")   # 없으면 자동 다운로드 (~6 MB)
    print("[SafeStep] 신호등 탐지 모델 로드 완료 ✅")
except Exception as e:
    print(f"[SafeStep] 신호등 모델 로드 실패 (신호등 기능 비활성화): {e}")

# 신호등 색상 히스토리 (blinking 판별용, 최근 6프레임)
_light_history: deque = deque(maxlen=6)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 거리 추정 모델 (Depth-Anything-V2-Metric-Small)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
depth_pipe   = None
DEPTH_SCALE  = 0.65          # VisionGuide 캘리브레이션 계수

# 백그라운드 스레드 — depth_pipe 가 느려도 /detect 를 막지 않음
_depth_queue : queue.Queue        = queue.Queue(maxsize=1)
_depth_cache : np.ndarray | None  = None   # (H, W) float32, 단위 m
_depth_lock  : threading.Lock     = threading.Lock()

try:
    from transformers import pipeline as hf_pipeline
    print("[SafeStep] 거리 추정 모델 로드 중 (Depth-Anything-V2-Small)...")
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    )
    print("[SafeStep] 거리 추정 모델 로드 완료 ✅")
except Exception as e:
    print(f"[SafeStep] 거리 추정 모델 로드 실패 (거리 기능 비활성화): {e}")


def _depth_worker() -> None:
    """백그라운드에서 depth map 을 계속 갱신하는 워커 스레드."""
    global _depth_cache
    while True:
        try:
            img: Image.Image = _depth_queue.get(timeout=1.0)
            result    = depth_pipe(img)
            depth_np  = result["predicted_depth"].squeeze().cpu().numpy()  # (H, W)
            scaled    = (depth_np * DEPTH_SCALE).astype(np.float32)
            with _depth_lock:
                _depth_cache = scaled
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[SafeStep] 거리 추정 실패: {e}")


if depth_pipe is not None:
    _depth_thread = threading.Thread(target=_depth_worker, daemon=True, name="depth-worker")
    _depth_thread.start()
    print("[SafeStep] 거리 추정 워커 스레드 시작 ✅")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Depth 자동 ON/OFF (VisionGuide 방식)
# 추론 시간이 지속적으로 높으면 depth 자동 비활성화 → 서버 부하 완화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_depth_auto_disabled  = False
_depth_recent_times: list[float] = []   # 최근 depth 추론 시간 (ms)
_DEPTH_AUTO_OFF_MS    = 250.0   # 5회 평균 > 250ms → 자동 OFF
_DEPTH_AUTO_ON_MS     = 150.0   # 평균 < 150ms → 자동 복구
_DEPTH_WINDOW         = 5
_depth_off_until      = 0.0     # 자동 OFF 후 재시도 가능 시각 (epoch)
_DEPTH_RECHECK_SEC    = 60.0    # OFF 후 60초 뒤 재시도


def _update_depth_auto(elapsed_ms: float) -> None:
    """depth 추론 시간을 기록하고 자동 ON/OFF 결정."""
    global _depth_auto_disabled, _depth_off_until, _depth_recent_times
    _depth_recent_times.append(elapsed_ms)
    if len(_depth_recent_times) > _DEPTH_WINDOW:
        _depth_recent_times.pop(0)
    if len(_depth_recent_times) < _DEPTH_WINDOW:
        return
    avg = sum(_depth_recent_times) / _DEPTH_WINDOW
    now = time.time()
    if not _depth_auto_disabled and avg > _DEPTH_AUTO_OFF_MS:
        _depth_auto_disabled = True
        _depth_off_until = now + _DEPTH_RECHECK_SEC
        print(f"[SafeStep] ⚠️  Depth 자동 OFF — 평균 추론 {avg:.0f}ms > {_DEPTH_AUTO_OFF_MS:.0f}ms")
    elif _depth_auto_disabled and now >= _depth_off_until and avg <= _DEPTH_AUTO_ON_MS:
        _depth_auto_disabled = False
        _depth_recent_times.clear()
        print(f"[SafeStep] ✅ Depth 자동 복구 — 평균 추론 {avg:.0f}ms")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 프레임 캐시 (VisionGuide 방식)
# /detect 와 /segment 가 같은 프레임을 따로 보낼 때 한 번만 추론
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_detect_cache:  dict = {}   # {md5: (result_dict, timestamp)}
_segment_cache: dict = {}
_CACHE_TTL = 2.0            # 2초 이내 동일 이미지 → 캐시 재사용


def _cache_get(cache: dict, md5: str) -> dict | None:
    entry = cache.get(md5)
    if entry and time.time() - entry[1] < _CACHE_TTL:
        return entry[0]
    return None


def _cache_set(cache: dict, md5: str, result: dict) -> None:
    cache[md5] = (result, time.time())
    # 만료 항목 정리
    now = time.time()
    expired = [k for k, (_, t) in cache.items() if now - t > _CACHE_TTL * 2]
    for k in expired:
        del cache[k]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 클래스 → 카테고리 매핑
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SURFACE_CATEGORY = {
    # 보도 (안전)
    "sidewalk_asphalt":             "sidewalk",
    "sidewalk_blocks":              "sidewalk",
    "sidewalk_cement":              "sidewalk",
    "sidewalk_other":               "sidewalk",
    "sidewalk_soil_stone":          "sidewalk",
    "sidewalk_urethane":            "sidewalk",
    "braille_guide_blocks_normal":  "sidewalk",   # 점자 유도블록 (정상)
    # 차도 (위험)
    "roadway_normal":               "road",
    # 횡단보도 (주의)
    "roadway_crosswalk":            "crosswalk",
    "alley_crosswalk":              "crosswalk",
    # 골목 (주의)
    "alley_normal":                 "alley",
    "alley_speed_bump":             "alley",
    "bike_lane":                    "alley",
    # 위험 구역 (즉시 경고)
    "sidewalk_damaged":             "caution",
    "alley_damaged":                "caution",
    "braille_guide_blocks_damaged": "caution",
    "caution_zone_grating":         "caution",
    "caution_zone_manhole":         "caution",
    "caution_zone_repair_zone":     "caution",
    "caution_zone_stairs":          "caution",
    "caution_zone_tree_zone":       "caution",
}

# 카테고리 정수 인코딩 (픽셀 마스크 연산용)
CAT_INT = {"sidewalk": 1, "road": 2, "crosswalk": 3, "alley": 4, "caution": 5}

# RGBA 색상
COLOR_MAP = {
    "sidewalk":  (50,  200, 80,  160),   # 🟢 초록  — 보도
    "road":      (220, 50,  50,  160),   # 🔴 빨강  — 차도
    "crosswalk": (230, 180, 30,  170),   # 🟡 노랑  — 횡단보도
    "alley":     (230, 180, 30,  120),   # 🟡 연노랑 — 골목
    "caution":   (255, 120, 0,   180),   # 🟠 주황  — 위험구역
}

KOREAN_LABELS = {
    "person": "사람", "bicycle": "자전거", "car": "자동차", "truck": "트럭",
    "bus": "버스", "motorcycle": "오토바이", "scooter": "스쿠터",
    "wheelchair": "휠체어", "stroller": "유모차",
    "bollard": "안전봉",                        # 볼라드 → 안전봉
    "barricade": "바리케이드", "pole": "기둥", "kiosk": "키오스크",
    "movable_signage": "이동식 간판", "chair": "의자", "potted_plant": "화분",
    "carrier": "카트", "fire_hydrant": "소화전", "parking_meter": "주차 장치",
    "power_controller": "전력 장치", "dog": "개", "cat": "고양이",
    "tree_trunk": "나무",                       # 누락 클래스 추가
    "traffic_light": "신호등",
    "traffic_light_controller": "신호등 제어기",
    "traffic_sign": "표지판",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 접근 속도 계산 (VisionGuide approach-speed 방식)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 객체 클래스 → 그룹 매핑 (VisionGuide 확장판)
_GROUP_MAP: dict[str, str] = {
    # 차량
    "car":        "vehicle", "truck":      "vehicle",
    "bus":        "vehicle", "motorcycle": "vehicle",
    # 개인이동장치
    "scooter":    "micro",   "bicycle":    "micro",
    # 사람 / 이동체
    "person":     "person",  "stroller":   "person",
    "wheelchair": "person",  "carrier":    "person",
    "movable_signage": "person",
    # 고정 장애물
    "bollard":    "fixed",   "barricade":  "fixed",
    "bench":      "fixed",   "chair":      "fixed",
    "fire_hydrant": "fixed", "kiosk":      "fixed",
    "pole":       "fixed",   "potted_plant": "fixed",
    "tree_trunk": "fixed",   "parking_meter": "fixed",
    "power_controller": "fixed", "stop":   "fixed",
    "traffic_light_controller": "fixed",
    # 신호
    "traffic_light": "signal", "traffic_sign": "signal",
}

# 그룹별 접근 속도 임계값 (m/s, 이 이상이어야 위험 경고)
_GROUP_THRESH: dict[str, float] = {
    "vehicle": 0.3,   # 차량: 0.3 m/s 이상 접근 시 경고
    "micro":   1.2,   # 자전거·킥보드
    "person":  2.0,   # 사람
}

# 그룹별 이전 상태: {group: (min_depth_m, monotonic_time)}
_group_state: dict[str, tuple[float, float]] = {}


def _calc_approach_speed(group: str, min_depth: float) -> float | None:
    """
    그룹의 최소 depth 변화로 접근 속도 계산.
    양수 = 접근 중 (m/s), 음수 = 멀어지는 중.
    첫 호출 또는 샘플 간격이 비정상이면 None 반환.
    """
    now = time.monotonic()
    if group not in _group_state:
        _group_state[group] = (min_depth, now)
        return None
    prev_depth, prev_time = _group_state[group]
    dt = now - prev_time
    _group_state[group] = (min_depth, now)
    if dt < 0.05 or dt > 5.0:    # 너무 빠르거나 오래된 샘플
        return None
    speed = (prev_depth - min_depth) / dt   # 감소 = 접근 = 양수
    return round(speed, 3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 신호등 HSV 색상 분석 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_TRAFFIC_LIGHT_CLASS = 9   # COCO class id


def _detect_light_color(img_bgr: np.ndarray,
                        x1: int, y1: int, x2: int, y2: int) -> str | None:
    """
    신호등 바운딩박스 내에서 HSV로 빨강/초록 판별.
    등 위치는 상단 절반만 사용(하단은 하우징 잡음).
    """
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(w_img, x2)
    # 상단 60% 만 사용
    y2_crop = min(h_img, y1 + int((y2 - y1) * 0.60))
    crop = img_bgr[y1:y2_crop, x1:x2]
    if crop.size == 0:
        return None

    hsv      = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    min_px   = max(crop.shape[0] * crop.shape[1] * 0.03, 30)  # 3% or 30px

    red1     = cv2.inRange(hsv, (0,   50, 150), (10,  255, 255))
    red2     = cv2.inRange(hsv, (160, 50, 150), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    grn_mask = cv2.inRange(hsv, (40,  50, 150), (100, 255, 255))

    red_px = float(np.sum(red_mask)) / 255
    grn_px = float(np.sum(grn_mask)) / 255

    if red_px > min_px and red_px >= grn_px:
        return "red"
    if grn_px > min_px:
        return "green"
    return None


def _traffic_light_status() -> str | None:
    """
    최근 6프레임 히스토리로 최종 상태 결정.
      - green ≥ 2 AND None ≥ 2  →  "blinking"
      - red   ≥ 3               →  "red"
      - green ≥ 3               →  "green"
      - 그 외 가장 최근 non-None  →  그 값
    """
    if not _light_history:
        return None
    red_n   = sum(1 for c in _light_history if c == "red")
    grn_n   = sum(1 for c in _light_history if c == "green")
    none_n  = sum(1 for c in _light_history if c is None)
    if grn_n >= 2 and none_n >= 2:
        return "blinking"
    if red_n >= 3:
        return "red"
    if grn_n >= 3:
        return "green"
    for c in reversed(_light_history):
        if c is not None:
            return c
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = FastAPI(title="SafeStep Server")

# CORS — ngrok / 웹 클라이언트 모두 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    """서버 시작 시 ngrok 터널 자동 개설 (NGROK_AUTH_TOKEN 설정된 경우)."""
    try:
        from pyngrok import ngrok
        auth_token = os.getenv("NGROK_AUTH_TOKEN", "")
        domain     = os.getenv("NGROK_DOMAIN", "")
        if not auth_token or auth_token == "여기에_토큰_입력":
            print("[SafeStep] ngrok 미설정 — server/.env 에 NGROK_AUTH_TOKEN 을 추가하면 와이파이 없이 사용 가능")
            return
        ngrok.set_auth_token(auth_token)
        tunnel = ngrok.connect(8000, domain=domain) if domain else ngrok.connect(8000)
        pub = tunnel.public_url
        print(f"\n{'='*60}")
        print(f"[SafeStep] 📱 공개 URL (ngrok): {pub}")
        print(f"[SafeStep] RemoteDetector.kt 에 아래 URL 입력:")
        print(f"  SERVER_URL = \"{pub}/detect\"")
        print(f"{'='*60}\n")
    except ImportError:
        print("[SafeStep] pyngrok 미설치 — pip install pyngrok 으로 설치 후 재시작")
    except Exception as e:
        print(f"[SafeStep] ngrok 실행 실패: {e}")


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "yolo":         MODEL_PATH.name,
        "surface":      SURFACE_PATH.name if surface_model else None,
        "signal":       "yolov8n" if signal_model else None,
        "depth":        "Depth-Anything-V2-Small" if depth_pipe else None,
        "depth_active": depth_pipe is not None and not _depth_auto_disabled,
        "depth_auto_off": _depth_auto_disabled,
    }


@app.post("/fast")
async def detect_fast(file: UploadFile = File(...)):
    """
    경량 고속 탐지 — 차량·사람 전용 (imgsz=160, depth 없음).
    매 프레임 호출해 긴급 충돌 경고 반응속도 극대화.
    """
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"이미지 파싱 실패: {e}")

    w, h = image.size
    results = yolo_model(image, imgsz=160, verbose=False, half=_half)

    _FAST_GROUPS = {"vehicle", "micro", "person"}
    detections = []
    for result in results:
        for box in result.boxes:
            label = yolo_model.names[int(box.cls[0])]
            grp   = _GROUP_MAP.get(label)
            if grp not in _FAST_GROUPS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            nx1, ny1 = x1 / w, y1 / h
            nx2, ny2 = x2 / w, y2 / h
            detections.append({
                "label":         label,
                "label_ko":      KOREAN_LABELS.get(label, label),
                "confidence":    round(conf, 3),
                "box":           [round(nx1,4), round(ny1,4), round(nx2,4), round(ny2,4)],
                "cx":            round((nx1+nx2)/2, 4),
                "area":          round((nx2-nx1)*(ny2-ny1), 4),
                "depth_m":       None,
                "group":         grp,
                "approach_speed": None,
            })
    detections.sort(key=lambda d: d["area"], reverse=True)
    return JSONResponse({"detections": detections})


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    raw = await file.read()

    # ── 프레임 캐시 확인 ─────────────────────────────────────────────────────
    md5 = hashlib.md5(raw).hexdigest()
    cached_result = _cache_get(_detect_cache, md5)
    if cached_result is not None:
        return JSONResponse(cached_result)

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"이미지 파싱 실패: {e}")

    w, h    = image.size
    results = yolo_model(image, imgsz=288, verbose=False, half=_half)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf  = float(box.conf[0])
            label = yolo_model.names[int(box.cls[0])]
            nx1, ny1 = x1 / w, y1 / h
            nx2, ny2 = x2 / w, y2 / h
            detections.append({
                "label":      label,
                "label_ko":   KOREAN_LABELS.get(label, label),
                "confidence": round(conf, 3),
                "box":        [round(nx1,4), round(ny1,4), round(nx2,4), round(ny2,4)],
                "cx":         round((nx1+nx2)/2, 4),
                "area":       round((nx2-nx1)*(ny2-ny1), 4),
                "depth_m":    None,  # 아래에서 채움
            })
    detections.sort(key=lambda d: d["area"], reverse=True)

    # ── 깊이 추정 ─────────────────────────────────────────────────────────────
    # 자동 OFF 상태면 depth 건너뜀 → 서버 부하 완화
    if depth_pipe is not None and not _depth_auto_disabled:
        t0 = time.time()
        try:
            _depth_queue.put_nowait(image)
        except queue.Full:
            pass  # 워커가 아직 처리 중 — 이전 캐시 사용
        elapsed_ms = (time.time() - t0) * 1000
        _update_depth_auto(elapsed_ms)

    # 현재 캐시된 depth map 으로 각 탐지 객체 거리 계산
    with _depth_lock:
        depth_snap = _depth_cache

    if depth_snap is not None:
        dh, dw = depth_snap.shape
        for det in detections:
            box  = det["box"]            # [nx1, ny1, nx2, ny2]
            nx1, ny1, nx2, ny2 = box

            # 박스 하단 30% × 가로 50% ROI (발 근처 — 배경/하늘 노이즈 제거)
            roi_y1 = ny1 + (ny2 - ny1) * 0.70   # 하단 30% 시작
            roi_x1 = nx1 + (nx2 - nx1) * 0.25   # 가로 중앙 50%
            roi_x2 = nx1 + (nx2 - nx1) * 0.75

            py1 = max(0, min(int(roi_y1 * dh), dh - 1))
            py2 = max(0, min(int(ny2   * dh), dh))
            px1 = max(0, min(int(roi_x1 * dw), dw - 1))
            px2 = max(0, min(int(roi_x2 * dw), dw))

            roi = depth_snap[py1:py2, px1:px2]
            if roi.size == 0:
                # ROI가 너무 작으면 중심점 fallback
                cy_px = min(int(((ny1 + ny2) / 2) * dh), dh - 1)
                cx_px = min(int(det["cx"] * dw), dw - 1)
                roi = depth_snap[max(0,cy_px-1):cy_px+2, max(0,cx_px-1):cx_px+2]

            # 25th percentile — 가장 가까운 값 위주로 (이상치/배경 제거)
            val = float(np.percentile(roi, 25)) if roi.size > 0 else 0.0
            det["depth_m"] = round(val, 2)

    # ── 접근 속도 계산 ────────────────────────────────────────────────────────
    groups_min: dict[str, float] = {}
    for det in detections:
        grp = _GROUP_MAP.get(det["label"])
        if grp and det["depth_m"] is not None:
            if grp not in groups_min or det["depth_m"] < groups_min[grp]:
                groups_min[grp] = det["depth_m"]

    groups_speed: dict[str, float | None] = {
        grp: _calc_approach_speed(grp, min_d)
        for grp, min_d in groups_min.items()
    }

    for det in detections:
        grp = _GROUP_MAP.get(det["label"])
        det["group"]          = grp
        det["approach_speed"] = groups_speed.get(grp) if grp else None

    resp = {"detections": detections}
    _cache_set(_detect_cache, md5, resp)
    return JSONResponse(resp)


@app.post("/signal")
async def signal_detect(file: UploadFile = File(...)):
    """
    신호등 색상 감지.
    yolov8n 으로 traffic_light 박스 탐지 → HSV 분석 → blinking 판별.
    Returns: {"color": "red" | "green" | "blinking" | null, "confidence": float | null}
    """
    if signal_model is None:
        return JSONResponse({"color": None, "confidence": None})

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"이미지 파싱 실패: {e}")

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    w, h    = image.size

    # traffic_light(class 9) 만 탐지
    results    = signal_model(image, classes=[_TRAFFIC_LIGHT_CLASS],
                              imgsz=320, conf=0.35, verbose=False, half=_half)
    best_color : str | None = None
    best_conf  : float      = 0.0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < best_conf:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            color = _detect_light_color(img_bgr, x1, y1, x2, y2)
            if color is not None:
                best_color = color
                best_conf  = conf

    _light_history.append(best_color)
    final_color = _traffic_light_status()

    return JSONResponse({
        "color":      final_color,
        "confidence": round(best_conf, 3) if best_color else None,
    })


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    if surface_model is None:
        raise HTTPException(503, "surface.pt 가 없습니다. 서버 폴더에 복사 후 재시작해주세요.")

    raw = await file.read()

    # ── 프레임 캐시 확인 ─────────────────────────────────────────────────────
    md5 = hashlib.md5(raw).hexdigest()
    cached_result = _cache_get(_segment_cache, md5)
    if cached_result is not None:
        return JSONResponse(cached_result)

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"이미지 파싱 실패: {e}")

    orig_w, orig_h = image.size

    # ── YOLO 세그멘테이션 추론 ─────────────────────────────
    results = surface_model(image, imgsz=512, verbose=False, half=_half)
    result  = results[0]

    # ── 픽셀 마스크 생성 ────────────────────────────────────
    cat_mask  = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)

    if result.masks is not None:
        seg_masks = result.masks.data.cpu().numpy()   # (N, mH, mW)
        cls_arr   = result.boxes.cls.cpu().numpy().astype(int)
        conf_arr  = result.boxes.conf.cpu().numpy()

        # 신뢰도 낮은 것부터 먼저 그려 높은 것이 위에 덮이게
        order = np.argsort(conf_arr)
        for i in order:
            if conf_arr[i] < 0.25:
                continue
            cls_name = surface_model.names[cls_arr[i]]
            category = SURFACE_CATEGORY.get(cls_name)
            if category is None:
                continue

            # 마스크를 원본 크기로 리사이즈
            mask_bin = cv2.resize(
                seg_masks[i].astype(np.float32), (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            ) > 0.5

            cat_mask[mask_bin]  = CAT_INT[category]
            mask_rgba[mask_bin] = COLOR_MAP[category]

    # ── 중앙 하단 1/3 구역으로 전체 상태 판단 ─────────────
    cy0       = orig_h * 2 // 3
    cx0, cx1  = orig_w // 3, orig_w * 2 // 3
    region    = cat_mask[cy0:, cx0:cx1]

    counts = {cat: int(np.sum(region == ci)) for cat, ci in CAT_INT.items()}
    total  = max(sum(counts.values()), 1)
    ratios = {k: round(v / total, 3) for k, v in counts.items()}

    if counts["caution"] > total * 0.05:
        status = "caution"
    elif counts["road"] > total * 0.10:
        status = "road"
    else:
        dominant = max(counts, key=counts.get)
        status   = dominant if counts[dominant] > 0 else "unknown"

    # ── 3구역 분석 (left / center / right) ────────────────
    # 하단 1/3 전체를 가로 3등분해서 각 구역 노면 상태 판정
    zone_cols = {
        "left":   (0,               orig_w // 3),
        "center": (orig_w // 3,     orig_w * 2 // 3),
        "right":  (orig_w * 2 // 3, orig_w),
    }
    zones: dict[str, str] = {}
    for z_name, (zx0, zx1) in zone_cols.items():
        zr      = cat_mask[cy0:, zx0:zx1]
        zc      = {cat: int(np.sum(zr == ci)) for cat, ci in CAT_INT.items()}
        zt      = max(sum(zc.values()), 1)
        if zc["caution"] > zt * 0.05:
            zones[z_name] = "caution"
        elif zc["road"] > zt * 0.10:
            zones[z_name] = "road"
        else:
            dom = max(zc, key=zc.get)
            zones[z_name] = dom if zc[dom] > 0 else "unknown"

    # ── 계단 감지 — 중앙 하단에 caution_zone_stairs 가 있으면 True ───────────
    is_stairs = False
    if result.masks is not None:
        for i in order:
            cls_name = surface_model.names[cls_arr[i]]
            if "stairs" in cls_name and conf_arr[i] >= 0.25:
                mask_bin = cv2.resize(
                    seg_masks[i].astype(np.float32), (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR
                ) > 0.5
                # 중앙 하단 구역에 200px 이상이면 계단으로 판정
                if np.sum(mask_bin[cy0:, cx0:cx1]) > 200:
                    is_stairs = True
                    break

    # ── PNG → base64 ───────────────────────────────────────
    pil_mask = Image.fromarray(mask_rgba, mode="RGBA")
    buf = io.BytesIO()
    pil_mask.save(buf, format="PNG", optimize=True)
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    resp = {
        "mask_b64":  mask_b64,
        "status":    status,
        "ratios":    ratios,
        "zones":     zones,
        "is_stairs": is_stairs,
    }
    _cache_set(_segment_cache, md5, resp)
    return JSONResponse(resp)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print("[SafeStep] 서버 시작 (포트 8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)

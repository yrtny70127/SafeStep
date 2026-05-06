"""
두 YOLO 모델(노면 세그멘테이션 + 장애물 탐지)을 로드하고
한 프레임을 분석해서 결과를 반환하는 서비스
"""
import base64
import time
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

from backend.utils.logger import get_logger

logger = get_logger(__name__)

SURFACE_MODEL_PATH = "ai_models/yolo_surface/weights/best.pt"
OBSTACLE_MODEL_PATH = "ai_models/yolo_obstacle/weights/best.pt"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

_surface_model = None
_obstacle_model = None
_depth_pipe = None

# 모든 라우터가 공유하는 단일 executor (모델은 thread-safe하지 않으므로 1개만)
shared_executor = ThreadPoolExecutor(max_workers=1)

# 프레임 간 거리 추적: cls_name → (depth_m, timestamp)
_prev_depths: dict[str, tuple[float, float]] = {}

# 신호등 색상 히스토리 (깜빡임 감지용, 최근 6프레임)
_light_history: list[str | None] = []

# Ollama
_ollama_enabled = False    # 대시보드 토글로 제어
_last_llm_call = 0.0
_LLM_MIN_INTERVAL = 3.0   # 최소 3초 간격 (너무 자주 호출 방지)

# Depth
_depth_enabled = True      # 대시보드 토글로 제어 (CPU 환경에서 끄면 속도 향상)

# ── 서버 부하 기반 Depth 자동 ON/OFF ─────────────────────────────────────
_depth_auto_disabled = False           # 자동 OFF 상태 (수동 OFF와 별개)
_depth_recent_times: list[float] = []  # 최근 추론 시간(ms) 슬라이딩 윈도우
_DEPTH_AUTO_OFF_MS  = 250.0            # 5회 평균 250ms 초과 → OFF
_DEPTH_AUTO_ON_MS   = 150.0            # 한 번 OFF 됐을 때 180ms 이하면 다시 시도
_DEPTH_WINDOW       = 5
_depth_off_until    = 0.0              # OFF 후 재시도 가능 시각 (초 단위 epoch)
_DEPTH_RECHECK_SEC  = 60.0             # OFF 후 60초 뒤에 짧게 다시 켜봄


def get_depth_status() -> dict:
    """클라이언트에 보내줄 Depth 상태 (auto_off / reason)"""
    return {
        "enabled":      _depth_enabled and not _depth_auto_disabled,
        "auto_off":     _depth_auto_disabled,
        "manual_off":   not _depth_enabled,
        "reason":       "server_overload" if _depth_auto_disabled else "",
    }


def set_ollama_enabled(value: bool):
    global _ollama_enabled
    _ollama_enabled = value


def get_ollama_enabled() -> bool:
    return _ollama_enabled


def set_depth_enabled(value: bool):
    global _depth_enabled
    _depth_enabled = value


def get_depth_enabled() -> bool:
    return _depth_enabled

# 그룹별 접근 위험 기준 (m/s)
# vehicle: 조금만 가까워져도 위험
# micro  : 빠르게 오면 위험, 느리면 내가 걷는 것
# person : 매우 빠르게 오면 경고
# fixed  : 거리 기반 (속도 무관), 가까워지면 1회 경고
# 깊이 보정 배수 (모델이 실제보다 멀게 측정하는 경향 보정)
DEPTH_SCALE = 0.65

_APPROACH_SPEED_THRESHOLD = {
    "vehicle": 0.3,   # 0.3m/s 이상 접근 → 위험
    "micro":   1.2,   # 1.2m/s 이상 접근 → 위험 (자전거 정상 주행 ~4m/s)
    "person":  2.0,   # 2.0m/s 이상 접근 → 경고 (뛰어오는 경우)
    "fixed":   None,  # 속도 무관, 거리로만 판단
    "signal":  None,  # 경고 없음
    "other":   None,
}


# ============================================================
# 클래스 → 그룹 매핑 (test 파일과 동일)
# ============================================================
OBSTACLE_CLASS_TO_GROUP = {
    "car": "vehicle", "bus": "vehicle", "truck": "vehicle", "motorcycle": "vehicle",
    "bicycle": "micro", "scooter": "micro",
    "person": "person", "cat": "person", "dog": "person",
    "stroller": "person", "wheelchair": "person", "carrier": "person",
    "movable_signage": "person",
    "barricade": "fixed", "bench": "fixed", "bollard": "fixed",
    "chair": "fixed", "fire_hydrant": "fixed", "kiosk": "fixed",
    "parking_meter": "fixed", "pole": "fixed", "potted_plant": "fixed",
    "power_controller": "fixed", "stop": "fixed", "table": "fixed",
    "tree_trunk": "fixed", "traffic_light_controller": "fixed",
    "traffic_light": "signal", "traffic_sign": "signal",
}

OBSTACLE_GROUP_WEIGHT = {
    "vehicle": 10, "micro": 9, "person": 6,
    "fixed": 4, "signal": 2, "other": 1,
}

SURFACE_CLASS_TO_GROUP = {
    "roadway_normal": "danger", "roadway_crosswalk": "danger",
    "alley_normal": "caution", "alley_crosswalk": "caution",
    "alley_damaged": "caution", "alley_speed_bump": "caution",
    "sidewalk_asphalt": "safe", "sidewalk_blocks": "safe",
    "sidewalk_cement": "safe", "sidewalk_urethane": "safe",
    "sidewalk_soil_stone": "safe", "sidewalk_other": "safe",
    "sidewalk_damaged": "caution",
    "braille_guide_blocks_normal": "braille",
    "braille_guide_blocks_damaged": "caution",
    "caution_zone_grating": "caution", "caution_zone_manhole": "caution",
    "caution_zone_repair_zone": "caution", "caution_zone_stairs": "caution",
    "caution_zone_tree_zone": "caution",
    "bike_lane": "bike",
}

KR_LABEL = {
    "car": "승용차", "bus": "버스", "truck": "트럭", "motorcycle": "오토바이",
    "bicycle": "자전거", "scooter": "스쿠터",
    "person": "사람", "cat": "고양이", "dog": "개",
    "stroller": "유모차", "wheelchair": "휠체어", "carrier": "리어카",
    "movable_signage": "이동식 안내판",
    "barricade": "바리케이드", "bench": "벤치", "bollard": "볼라드",
    "fire_hydrant": "소화전", "kiosk": "키오스크", "pole": "기둥",
    "tree_trunk": "가로수", "traffic_light": "신호등", "traffic_sign": "교통표지판",
    "stop": "정류장",
    "roadway_normal": "차도", "roadway_crosswalk": "차도 횡단보도",
    "alley_normal": "골목길", "alley_speed_bump": "과속방지턱",
    "sidewalk_asphalt": "인도", "sidewalk_blocks": "인도",
    "sidewalk_cement": "인도", "sidewalk_urethane": "인도",
    "sidewalk_soil_stone": "인도", "sidewalk_other": "인도",
    "sidewalk_damaged": "파손 인도",
    "braille_guide_blocks_normal": "점자블록",
    "braille_guide_blocks_damaged": "파손 점자블록",
    "caution_zone_manhole": "맨홀", "caution_zone_stairs": "계단",
    "caution_zone_repair_zone": "공사 구역",
    "bike_lane": "자전거 도로",
}


# ============================================================
# 모델 로드 (첫 호출 시 한 번만)
# ============================================================
def _load_models():
    global _surface_model, _obstacle_model, _depth_pipe

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics 패키지가 없습니다. pip install ultralytics")

    if _surface_model is None:
        path = Path(SURFACE_MODEL_PATH)
        if path.exists():
            logger.info(f"노면 모델 로드: {path}")
            _surface_model = YOLO(str(path))
        else:
            logger.warning(f"노면 모델 없음: {path}")

    if _obstacle_model is None:
        path = Path(OBSTACLE_MODEL_PATH)
        if path.exists():
            logger.info(f"장애물 모델 로드: {path}")
            _obstacle_model = YOLO(str(path))
        else:
            logger.warning(f"장애물 모델 없음: {path}")

    if _depth_pipe is None:
        try:
            from transformers import pipeline as hf_pipeline
            device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if device == 0 else "CPU"
            logger.info(f"Depth-Anything-V2 로드 중 ({device_name}) ... 첫 실행 시 다운로드될 수 있습니다")
            _depth_pipe = hf_pipeline(
                task="depth-estimation",
                model=DEPTH_MODEL_ID,
                device=device,
            )
            logger.info("Depth-Anything-V2 로드 완료")
        except Exception as e:
            logger.warning(f"Depth 모델 로드 실패 (거리 추정 비활성화): {e}")


# ============================================================
# 신호등 색상 감지 (HSV)
# ============================================================
def _detect_traffic_light_color(frame: np.ndarray, bbox: list) -> str | None:
    """바운딩 박스 내 신호등 색상 판별. 'red' | 'green' | None
    야간/주간 모두 대응: 채도 기준 낮추고 밝은 영역 위주로 판별"""
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 빨간색 (채도/밝기 기준 완화 - 야간 대응)
    red1 = cv2.inRange(hsv, (0,   50, 150), (10,  255, 255))
    red2 = cv2.inRange(hsv, (160, 50, 150), (180, 255, 255))
    red_px = int(cv2.bitwise_or(red1, red2).sum())

    # 초록색 (채도/밝기 기준 완화)
    green_mask = cv2.inRange(hsv, (40, 50, 150), (100, 255, 255))
    green_px = int(green_mask.sum())

    # 최소 픽셀: 바운딩 박스 면적의 2% 이상이어야 유효
    min_px = max(crop.shape[0] * crop.shape[1] * 255 * 0.02, 50)
    if red_px > green_px and red_px > min_px:
        return "red"
    if green_px > red_px and green_px > min_px:
        return "green"
    return None


def _check_light_blinking(color: str | None) -> bool:
    """최근 프레임에서 초록불이 깜빡이는지 감지"""
    global _light_history
    _light_history.append(color)
    if len(_light_history) > 6:
        _light_history.pop(0)
    greens = _light_history.count("green")
    nones  = _light_history.count(None)
    # 초록과 None이 번갈아 나오면 깜빡임
    return greens >= 2 and nones >= 2


# ============================================================
# Ollama LLaMA 안내 메시지 생성
# ============================================================
def _generate_guidance_llm(situation: str) -> str | None:
    """Ollama LLaMA로 자연스러운 한국어 안내 메시지 생성. 실패 시 None 반환."""
    global _last_llm_call
    now = time.time()
    if now - _last_llm_call < _LLM_MIN_INTERVAL:
        return None  # 너무 자주 호출 방지

    import os, re
    model = os.getenv("OLLAMA_MODEL", "llama3.2")

    try:
        import ollama
        _last_llm_call = now
        resp = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 시각장애인을 위한 보행 안내 AI야.\n"
                        "아래 규칙을 반드시 지켜:\n\n"
                        "# 출력 규칙\n"
                        "- 15글자 이내, 딱 한 문장만\n"
                        "- 한국어만, 이모지/영어/특수문자 절대 금지\n"
                        "- 안전한 상황이면 아무것도 출력하지 마\n\n"
                        "# 우선순위 (높은 순)\n"
                        "1. 신호등 (빨간불 → 무조건 멈춰)\n"
                        "2. 차량/오토바이 접근\n"
                        "3. 자전거/스쿠터 접근\n"
                        "4. 사람 충돌 위험\n"
                        "5. 고정 장애물\n\n"
                        "# 출력 예시\n"
                        "- '왼쪽 차 접근 피하세요'\n"
                        "- '빨간불 멈추세요'\n"
                        "- '오른쪽으로 피하세요'\n"
                        "- '정면 볼라드 주의'\n"
                    ),
                },
                {
                    "role": "user",
                    "content": f"감지 정보:\n{situation}\n\n경고:",
                },
            ],
            options={"num_predict": 80, "temperature": 0.3},
        )
        msg = resp["message"]["content"].strip()
        msg = re.sub(r'[^\w\s.,!?%~\-가-힣]', '', msg).strip()
        return msg if msg else None
    except Exception as e:
        logger.warning(f"Ollama 호출 실패: {e}")
        return None


def _estimate_depth(frame: np.ndarray) -> np.ndarray | None:
    """BGR numpy array → metric depth map (HxW, float32, 단위: 미터). 실패 시 None."""
    global _depth_auto_disabled, _depth_off_until
    if _depth_pipe is None:
        return None
    # ── 자동 OFF 상태에서 재시도 시각 도래 전이면 스킵 ──
    now = time.time()
    if _depth_auto_disabled and now < _depth_off_until:
        return None

    t0 = time.time()
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        result = _depth_pipe(pil_img)
        # "depth"는 표시용 정규화 이미지, "predicted_depth"가 실제 미터값
        depth = result["predicted_depth"].squeeze().numpy().astype(np.float32)
        h, w = frame.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── 추론 시간 추적 + 자동 ON/OFF 판정 ──
        elapsed_ms = (time.time() - t0) * 1000.0
        _depth_recent_times.append(elapsed_ms)
        if len(_depth_recent_times) > _DEPTH_WINDOW:
            _depth_recent_times.pop(0)
        if len(_depth_recent_times) == _DEPTH_WINDOW:
            avg = sum(_depth_recent_times) / _DEPTH_WINDOW
            if not _depth_auto_disabled and avg > _DEPTH_AUTO_OFF_MS:
                _depth_auto_disabled = True
                _depth_off_until = now + _DEPTH_RECHECK_SEC
                logger.warning(f"[Depth 자동 OFF] 평균 {avg:.0f}ms > {_DEPTH_AUTO_OFF_MS}ms")
            elif _depth_auto_disabled and avg < _DEPTH_AUTO_ON_MS:
                _depth_auto_disabled = False
                _depth_recent_times.clear()
                logger.info(f"[Depth 자동 ON] 평균 {avg:.0f}ms < {_DEPTH_AUTO_ON_MS}ms")

        return depth * DEPTH_SCALE
    except Exception as e:
        logger.warning(f"깊이 추정 실패: {e}")
        return None


# ============================================================
# 장애물 분석
# ============================================================
def _analyze_obstacle(frame: np.ndarray, depth_map: np.ndarray | None = None, raw_frame: np.ndarray | None = None) -> dict:
    if _obstacle_model is None:
        return {"available": False}

    results = _obstacle_model.predict(frame, conf=0.35, imgsz=640, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return {"available": True, "top": None, "counts": {}}

    class_names = _obstacle_model.names
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    h, w = frame.shape[:2]
    items = []
    counts = {}
    class_counts = {}

    for box, cls_id, conf in zip(boxes, classes, confs):
        cls_name = class_names.get(cls_id, f"cls{cls_id}")
        group = OBSTACLE_CLASS_TO_GROUP.get(cls_name, "other")
        x1, y1, x2, y2 = box.astype(int)
        area_pct = float(((x2 - x1) * (y2 - y1)) / (h * w) * 100)
        center_y = int((y1 + y2) // 2)

        depth_m = None
        if depth_map is not None:
            # 바운딩 박스 하단 30% + 가로 중앙 50% 영역 사용
            # (배경 포함 줄이고 물체가 땅에 닿는 지점 기준)
            bh = y2 - y1
            bw = x2 - x1
            ry1 = y1 + int(bh * 0.7)
            rx1 = x1 + int(bw * 0.25)
            rx2 = x1 + int(bw * 0.75)
            roi = depth_map[ry1:y2, rx1:rx2]
            if roi.size > 0:
                # 하위 25% 백분위수 → 배경값 제거 후 가장 가까운 값
                depth_m = round(float(np.percentile(roi, 25)), 1)

        # 신호등 색상 감지
        light_color = None
        if cls_name == "traffic_light" and raw_frame is not None:
            light_color = _detect_traffic_light_color(raw_frame, [int(x1), int(y1), int(x2), int(y2)])

        # 방향 계산 (바운딩박스 중심 x / 프레임 너비)
        cx_ratio = ((x1 + x2) / 2) / w
        if cx_ratio < 0.33:
            direction = "왼쪽"
        elif cx_ratio < 0.66:
            direction = "정면"
        else:
            direction = "오른쪽"

        items.append({
            "cls_name": cls_name,
            "group": group,
            "conf": float(conf),
            "area_pct": area_pct,
            "center_y": center_y,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "depth_m": depth_m,
            "light_color": light_color,
            "direction": direction,
        })
        counts[group] = counts.get(group, 0) + 1
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    # 위험도 점수로 정렬
    def risk_score(item):
        weight = OBSTACLE_GROUP_WEIGHT.get(item["group"], 1)
        y_weight = (item["center_y"] / h) ** 2
        area_weight = min(item["area_pct"] / 20, 1.0)
        return weight * (y_weight + area_weight)

    items.sort(key=risk_score, reverse=True)
    return {"available": True, "top": items[0] if items else None, "counts": counts, "class_counts": class_counts, "items": items}


# ============================================================
# 노면 분석
# ============================================================
def _analyze_surface(frame: np.ndarray) -> dict:
    if _surface_model is None:
        return {"available": False}

    results = _surface_model.predict(frame, conf=0.35, imgsz=640, verbose=False)
    result = results[0]

    if result.masks is None or len(result.masks) == 0:
        return {"available": True, "foot_zone": None, "group_ratios": {}}

    class_names = _surface_model.names
    h, w = frame.shape[:2]

    # 발 앞 영역: 하단 1/3 중앙 1/2
    fy1, fy2 = int(h * 0.66), h
    fx1, fx2 = int(w * 0.25), int(w * 0.75)
    foot_area = (fy2 - fy1) * (fx2 - fx1)

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    foot_scores = {}
    group_pixels = {}
    class_pixels = {}
    masks_viz = []  # (binary_mask_HW, group) - 시각화용, broadcast 전 제거

    for mask, cls_id in zip(masks, classes):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        cls_name = class_names.get(cls_id, f"cls{cls_id}")
        group = SURFACE_CLASS_TO_GROUP.get(cls_name, "neutral")

        foot_region = mask_resized[fy1:fy2, fx1:fx2]
        foot_px = int((foot_region > 0.5).sum())
        foot_scores[cls_name] = foot_scores.get(cls_name, 0) + foot_px

        full_px = int((mask_resized > 0.5).sum())
        group_pixels[group] = group_pixels.get(group, 0) + full_px
        class_pixels[cls_name] = class_pixels.get(cls_name, 0) + full_px

        masks_viz.append((mask_resized > 0.5, group, cls_name))

    total_px = h * w
    group_ratios = {g: p / total_px * 100 for g, p in group_pixels.items()}
    class_ratios = {c: p / total_px * 100 for c, p in class_pixels.items()}

    # 3구역 분석 (왼쪽 / 정면 / 오른쪽) - 하단 1/3 기준
    zone_defs = {
        "왼쪽":   (0,          int(w * 0.33)),
        "정면":   (int(w * 0.33), int(w * 0.67)),
        "오른쪽": (int(w * 0.67), w),
    }
    zones = {}
    for zone_name, (zx1, zx2) in zone_defs.items():
        zone_scores = {}
        for mask_bin, grp, cls_name in masks_viz:
            zone_px = int(mask_bin[fy1:fy2, zx1:zx2].sum())
            if zone_px > 0:
                zone_scores[cls_name] = zone_scores.get(cls_name, 0) + zone_px
        if zone_scores:
            dom = max(zone_scores, key=zone_scores.get)
            g = SURFACE_CLASS_TO_GROUP.get(dom, "neutral")
            zones[zone_name] = {"cls_name": dom, "group": g, "kr": KR_LABEL.get(dom, dom)}
        else:
            zones[zone_name] = None

    if not foot_scores:
        return {"available": True, "foot_zone": None, "zones": zones,
                "group_ratios": group_ratios, "class_ratios": class_ratios, "_masks_viz": masks_viz}

    dominant_cls = max(foot_scores, key=foot_scores.get)
    dominant_pct = foot_scores[dominant_cls] / foot_area * 100
    dominant_group = SURFACE_CLASS_TO_GROUP.get(dominant_cls, "neutral")

    return {
        "available": True,
        "foot_zone": {"cls_name": dominant_cls, "group": dominant_group, "pct": dominant_pct},
        "zones": zones,
        "group_ratios": group_ratios,
        "class_ratios": class_ratios,
        "_masks_viz": masks_viz,
    }


# ============================================================
# 안내 메시지 생성
# ============================================================
def _build_detail(obstacle: dict, surface: dict) -> dict:
    """화면 표시용 판단 근거 생성 (TTS와 별개)"""
    # 전체 장애물 목록 (최대 5개, 위험도 순)
    obs_list = []
    for i, item in enumerate(obstacle.get("items", [])[:5]):
        kr = KR_LABEL.get(item["cls_name"], item["cls_name"])
        depth_m = item.get("depth_m")
        prev = _prev_depths.get(item["cls_name"])
        approaching = False
        if prev and depth_m is not None:
            prev_d, prev_t = prev
            dt = time.time() - prev_t
            if 0.1 < dt < 5.0:
                approaching = (prev_d - depth_m) / dt > 0.3
        obs_list.append({
            "rank": i + 1,
            "kr": kr,
            "direction": item.get("direction", "정면"),
            "depth_m": depth_m,
            "conf": round(item["conf"] * 100),
            "group": item["group"],
            "approaching": approaching,
        })

    # 노면 3구역
    zones = surface.get("zones", {})
    surface_zones = {}
    for zone_name, z in zones.items():
        if z:
            surface_zones[zone_name] = {"kr": z["kr"], "group": z["group"]}

    # 회피 방향
    dodge = ""
    if obstacle.get("top"):
        top = obstacle["top"]
        dodge = _dodge_hint(top.get("direction", "정면"), obstacle.get("items", [])[1:], zones)

    return {"obstacles": obs_list, "surface_zones": surface_zones, "dodge": dodge}


def _build_situation_text(obstacle: dict, surface: dict) -> str:
    """Ollama에 넘길 상황 요약 텍스트 생성 (전체 장애물 + 노면 3구역)"""
    lines = []

    # 노면 3구역
    zones = surface.get("zones", {})
    if zones:
        zone_strs = []
        for zone_name in ["왼쪽", "정면", "오른쪽"]:
            z = zones.get(zone_name)
            if z:
                zone_strs.append(f"{zone_name}={z['kr']}({z['group']})")
        if zone_strs:
            lines.append("노면: " + ", ".join(zone_strs))

    # 전체 장애물 목록 (최대 5개)
    items = obstacle.get("items", [])
    for i, item in enumerate(items[:5]):
        kr = KR_LABEL.get(item["cls_name"], item["cls_name"])
        depth_m = item.get("depth_m")
        direction = item.get("direction", "정면")
        light_color = item.get("light_color")
        approach = _prev_depths.get(item["cls_name"])

        desc = f"장애물{i+1}: {direction} {kr}"
        if depth_m is not None:
            desc += f", {depth_m}m"
        if light_color == "red":
            desc += ", 빨간불"
        elif light_color == "green":
            desc += ", 초록불"
        if approach and depth_m is not None:
            prev_d, _ = approach
            delta = prev_d - depth_m
            if delta > 0.3:
                desc += ", 접근 중"
            elif delta < -0.3:
                desc += ", 멀어지는 중"
        lines.append(desc)

    # 코드가 제안하는 회피 방향 (LLaMA가 검토)
    if obstacle.get("top"):
        top = obstacle["top"]
        dodge = _dodge_hint(top.get("direction", "정면"), items[1:], zones)
        lines.append(f"회피 제안: {dodge}")

    return "\n".join(lines) if lines else "특이사항 없음"


def _calc_approach_speed(cls_name: str, depth_m: float) -> float | None:
    """
    이전 프레임 거리와 비교해 접근 속도(m/s) 반환.
    양수 = 가까워지는 중, 음수 = 멀어지는 중.
    """
    now = time.time()
    if cls_name in _prev_depths:
        prev_depth, prev_time = _prev_depths[cls_name]
        dt = now - prev_time
        if 0.1 < dt < 5.0:   # 너무 짧거나 긴 간격 무시
            speed = (prev_depth - depth_m) / dt  # 양수 = 접근
            _prev_depths[cls_name] = (depth_m, now)
            return round(speed, 2)
    _prev_depths[cls_name] = (depth_m, now)
    return None


def _dodge_hint(direction: str, other_items: list | None = None, surface_zones: dict | None = None) -> str:
    """장애물 방향 + 주변 장애물 + 노면 안전 여부를 종합해 회피 방향 결정."""
    sz = surface_zones or {}

    def zone_danger(zone_name: str) -> bool:
        z = sz.get(zone_name)
        return z is not None and z.get("group") == "danger"

    def zone_safe(zone_name: str) -> bool:
        z = sz.get(zone_name)
        return z is None or z.get("group") in ("safe", "braille", "caution", "neutral")

    if direction == "왼쪽":
        # 장애물이 왼쪽 → 오른쪽으로 피함, 오른쪽이 차도면 멈춤
        if zone_danger("오른쪽"):
            return "모든 방향 위험, 멈추세요"
        return "오른쪽으로 피하세요"
    elif direction == "오른쪽":
        if zone_danger("왼쪽"):
            return "모든 방향 위험, 멈추세요"
        return "왼쪽으로 피하세요"
    else:
        # 정면 → 장애물 없고 노면 안전한 쪽 선택
        left_blocked  = any(i.get("direction") == "왼쪽"  for i in (other_items or []))
        right_blocked = any(i.get("direction") == "오른쪽" for i in (other_items or []))
        left_ok  = not left_blocked  and zone_safe("왼쪽")
        right_ok = not right_blocked and zone_safe("오른쪽")

        if left_ok and right_ok:
            return "왼쪽으로 피하세요"
        elif left_ok:
            return "왼쪽으로 피하세요"
        elif right_ok:
            return "오른쪽으로 피하세요"
        else:
            return "모든 방향 위험, 멈추세요"


def _make_guidance(obstacle: dict, surface: dict) -> str:
    parts = []

    # ── 노면 안내 (위험/주의만, 안전한 인도는 생략) ──
    if surface.get("available") and surface.get("foot_zone"):
        fz = surface["foot_zone"]
        group = fz["group"]
        kr = KR_LABEL.get(fz["cls_name"], fz["cls_name"])

        if group == "danger":
            parts.append(f"⚠️ 발 앞에 {kr}가 있습니다. 주의하세요.")
        elif group == "bike":
            parts.append("⚠️ 자전거 도로입니다. 조심하세요.")
        elif group == "caution":
            parts.append(f"발 앞에 {kr}이 있으니 조심하세요.")
        elif group == "braille":
            parts.append("점자블록 위에 있습니다.")
        # safe → 생략

    # ── 장애물 안내 (접근 속도 기반) ──────────
    if obstacle.get("available") and obstacle.get("top"):
        top = obstacle["top"]
        group = top["group"]
        kr = KR_LABEL.get(top["cls_name"], top["cls_name"])
        depth_m = top.get("depth_m")
        area = top["area_pct"]
        direction = top.get("direction", "정면")
        dist_str = f"{depth_m}m " if depth_m is not None else ""
        loc_str = f"{dist_str}{direction}에" if direction != "정면" else f"{dist_str}정면에"
        other_items = obstacle.get("items", [])[1:]
        surface_zones = surface.get("zones", {})
        dodge = _dodge_hint(direction, other_items, surface_zones)

        approach_speed = None
        if depth_m is not None:
            approach_speed = _calc_approach_speed(top["cls_name"], depth_m)

        # area_pct 기반 근접도 (속도 정보 없을 때 폴백)
        proximity = "매우 가까운 " if area > 15 else "가까운 " if area > 5 else ""

        is_side = direction != "정면"

        if group == "vehicle":
            threshold = _APPROACH_SPEED_THRESHOLD["vehicle"]
            if approach_speed is not None and approach_speed > threshold:
                # 빠르게 접근 중 → 피하라고 안내
                parts.append(f"🚨 {kr}가 {loc_str}서 빠르게 접근 중입니다! {dodge}!")
            elif is_side or (approach_speed is not None and approach_speed < -0.5):
                pass  # 옆에 있거나 멀어지는 중 → 생략
            else:
                # 정면에 있지만 속도 불명 → 존재만 알림, 피하라고는 안 함
                parts.append(f"⚠️ {loc_str} {proximity}{kr}가 있습니다.")

        elif group == "micro":
            threshold = _APPROACH_SPEED_THRESHOLD["micro"]
            if approach_speed is not None and approach_speed > threshold:
                parts.append(f"🚨 {kr}가 {loc_str}서 빠르게 접근 중입니다! {dodge}!")
            elif is_side or (approach_speed is not None and approach_speed < -0.5):
                pass
            else:
                parts.append(f"{loc_str} {proximity}{kr}가 있습니다.")

        elif group == "person":
            threshold = _APPROACH_SPEED_THRESHOLD["person"]
            if approach_speed is not None and approach_speed > threshold:
                parts.append(f"⚠️ {kr}가 {loc_str}서 빠르게 접근 중입니다. {dodge}.")
            elif is_side or (approach_speed is not None and approach_speed < -1.0):
                pass
            else:
                parts.append(f"{loc_str} {proximity}{kr}가 있습니다.")

        elif group == "fixed":
            if is_side:
                pass
            elif depth_m is not None and depth_m < 5.0:
                # 고정 장애물은 정면에 있으면 항상 피하라고 안내
                parts.append(f"⚠️ {loc_str} {kr}가 있습니다. {dodge}.")
            elif depth_m is None:
                parts.append(f"정면에 {kr}가 있습니다. {dodge}.")

        elif group == "signal":
            light_color = top.get("light_color")
            blinking = _check_light_blinking(light_color)

            # 차량이 횡단보도에 있는지 확인
            car_at_cross = any(
                i["group"] == "vehicle" and i.get("depth_m", 99) < 15
                for i in obstacle.get("items", [])
            )
            # 차량 접근 중인지
            car_approaching = any(
                i["group"] == "vehicle" and
                _prev_depths.get(i["cls_name"], (999,0))[0] - (i.get("depth_m") or 999) > 0.5
                for i in obstacle.get("items", [])
            )

            if light_color == "red":
                parts.insert(0, f"빨간불 멈추세요{dist_str}")
            elif blinking:
                # 초록불 깜빡임
                if car_at_cross:
                    parts.insert(0, "초록불 깜빡임, 차 있음 주의")
                else:
                    parts.insert(0, "초록불 깜빡임, 서두르지 마세요")
            elif light_color == "green":
                if car_approaching:
                    parts.insert(0, f"초록불, 차량 접근 주의{dist_str}")
                elif car_at_cross:
                    parts.insert(0, f"초록불, 차 있음 확인 후 건너세요{dist_str}")
                else:
                    parts.insert(0, f"초록불 건너도 됩니다{dist_str}")
            else:
                parts.append(f"신호등 감지{dist_str}")

    if not parts:
        return "이상 없음"

    return " / ".join(parts)


# ============================================================
# 메인 함수
# ============================================================
# 그룹별 BGR 색상 (OpenCV는 BGR)
_GROUP_COLOR_BGR = {
    "vehicle": (68,  68,  255),
    "micro":   (102, 136, 255),
    "person":  (255, 179, 102),
    "fixed":   (68,  170, 255),
    "signal":  (77,  228, 255),
    "other":   (160, 160, 160),
}

# 노면 그룹 BGR 색상
_SURF_COLOR_BGR = {
    "safe":    (100, 220, 120),   # 초록
    "braille": (77,  228, 255),   # 노랑
    "caution": (68,  170, 255),   # 주황
    "danger":  (68,  68,  255),   # 빨강
    "bike":    (255, 124, 200),   # 보라
    "neutral": (128, 128, 128),
}


def _draw_boxes(frame: np.ndarray, obstacle: dict) -> np.ndarray:
    """장애물 바운딩 박스를 이미지에 그림."""
    items = obstacle.get("items", [])
    if not items:
        return frame
    annotated = frame.copy()
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        color = _GROUP_COLOR_BGR.get(item["group"], (160, 160, 160))
        label = KR_LABEL.get(item["cls_name"], item["cls_name"])
        conf_pct = int(item["conf"] * 100)

        # 반투명 채우기
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        annotated = cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0)

        # 테두리
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # 레이블 배경 + 텍스트
        text = f"{label} {conf_pct}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(annotated, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(annotated, text, (x1 + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


def analyze_frame(image_bytes: bytes) -> dict:
    """
    이미지 바이트를 받아 두 모델로 분석 후 결과 반환.
    camera_router.py에서 호출하는 진입점.
    """
    _load_models()

    # bytes → numpy array
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"message": "이미지 디코딩 실패", "error": True}

    depth_map = _estimate_depth(frame) if _depth_enabled else None
    obstacle = _analyze_obstacle(frame, depth_map, raw_frame=frame)
    surface = _analyze_surface(frame)
    rule_message = _make_guidance(obstacle, surface)

    # TTS 메시지 (Ollama ON이면 다듬기)
    message = rule_message
    if _ollama_enabled and rule_message != "이상 없음":
        situation = _build_situation_text(obstacle, surface)
        llm_msg = _generate_guidance_llm(situation)
        if llm_msg:
            message = llm_msg

    # 화면 표시용 판단 근거
    detail = _build_detail(obstacle, surface)

    h, w = frame.shape[:2]

    # ── 노면: 클래스별 반투명 BGRA PNG (클라이언트 하위 체크박스로 토글) ──
    masks_viz = surface.pop("_masks_viz", [])
    class_bgra = {}
    for binary_mask, group, cls_name in masks_viz:
        if cls_name not in class_bgra:
            class_bgra[cls_name] = np.zeros((h, w, 4), dtype=np.uint8)
        c = _SURF_COLOR_BGR.get(group, (128, 128, 128))
        class_bgra[cls_name][binary_mask] = [c[0], c[1], c[2], 140]  # BGRA, α=140

    surf_masks_b64 = {}
    for cls_name, bgra in class_bgra.items():
        _, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        surf_masks_b64[cls_name] = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

    # ── 장애물: bbox 리스트 (클라이언트에서 체크박스로 토글) ──
    obs_items = [
        {"bbox": item["bbox"], "group": item["group"],
         "cls_name": item["cls_name"], "conf": item["conf"],
         "depth_m": item.get("depth_m"),
         "direction": item.get("direction", "정면"),
         "light_color": item.get("light_color")}   # 신호등 색상 (red/green/None)
        for item in obstacle.get("items", [])
    ]

    # ── 기본 이미지 (원본, 오버레이 없음) ──
    _, raw_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    base_image_b64 = "data:image/jpeg;base64," + base64.b64encode(raw_buf.tobytes()).decode()

    obstacle_out = {k: v for k, v in obstacle.items() if k != "items"}

    return {
        "message": message,
        "detail": detail,
        "obstacle": obstacle_out,
        "surface": surface,
        "base_image": base_image_b64,
        "surface_masks": surf_masks_b64,
        "obstacle_items": obs_items,
        "img_width": w,
        "img_height": h,
        "depth_status": get_depth_status(),
    }

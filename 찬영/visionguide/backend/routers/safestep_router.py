"""
SafeStep 호환 API 엔드포인트
- POST /detect   : 장애물 탐지 (SafeStep RemoteDetector.kt 호환)
- POST /segment  : 노면 세그멘테이션 (SafeStep SegmentationClient.kt 호환)

SafeStep 앱은 /detect, /segment 를 같은 프레임으로 따로 요청함.
캐시를 통해 같은 이미지는 한 번만 추론하도록 처리.
"""
import asyncio
import hashlib
import time

from fastapi import APIRouter, UploadFile, File, Header, HTTPException, Request

from backend.services import inference_service, capture_service
from backend.services.inference_service import shared_executor
from backend.utils.logger import get_logger
from backend.routers.camera_router import update_latest_frame, dashboard_manager

logger = get_logger(__name__)
router = APIRouter()

# ── 캐시 + 진행 중 대기 (같은 프레임 2번 추론 방지) ──────────────────────
_cache: dict = {}      # {md5_hex: (result, timestamp)}
_seg_cache: dict = {}  # {md5_hex: (seg_result, timestamp)} - /segment 처리 결과 캐시
_pending: dict = {}    # {md5_hex: asyncio.Event} - 추론 진행 중인 키
_CACHE_TTL = 2.0       # 2초 이내 동일 이미지 → 재사용

# ── 동시 접속 제한 (슬롯 1, 2) ──────────────────────────────────────────
SLOTS = (1, 2)
MAX_CLIENTS = len(SLOTS)
SESSION_TTL = 30.0     # 30초 미요청 시 슬롯 해제
# {client_id: (slot, last_request_ts)}
_sessions: dict[str, tuple[int, float]] = {}


def _gc_sessions(now: float) -> list[int]:
    """만료된 세션 제거. 해제된 슬롯 번호 목록 반환."""
    freed: list[int] = []
    for cid, (slot, ts) in list(_sessions.items()):
        if now - ts > SESSION_TTL:
            _sessions.pop(cid, None)
            freed.append(slot)
            logger.info(f"[SafeStep 슬롯] {cid} 만료 → 슬롯 {slot} 해제")
    return freed


def _check_slot(client_id: str) -> int:
    """
    슬롯 가용성 확인 + 할당된 슬롯 번호(1 또는 2) 반환.
    - 이미 활성 세션이면 기존 슬롯 유지 + ts 갱신
    - 새 세션이고 빈 슬롯 있으면 가장 작은 번호 슬롯 할당
    - 빈 슬롯 없으면 429
    """
    now = time.time()
    freed = _gc_sessions(now)
    if freed:
        # 해제된 슬롯의 프레임 버퍼 비우기 (대시보드가 "비어있음"으로 표시)
        from backend.routers.camera_router import clear_slot_frame
        for s in freed:
            clear_slot_frame(s)

    if client_id in _sessions:
        slot, _ = _sessions[client_id]
        _sessions[client_id] = (slot, now)
        return slot

    used = {s for s, _ in _sessions.values()}
    for slot in SLOTS:
        if slot not in used:
            _sessions[client_id] = (slot, now)
            logger.info(f"[SafeStep 슬롯] 새 클라이언트 {client_id} → 슬롯 {slot}")
            return slot

    raise HTTPException(
        status_code=429,
        detail={"error": "server_busy", "message": f"최대 {MAX_CLIENTS}명까지만 동시 접속 가능합니다."},
    )


def get_active_slots() -> dict[int, str]:
    """현재 활성 슬롯 → client_id 매핑 (대시보드용)"""
    now = time.time()
    _gc_sessions(now)
    return {slot: cid for cid, (slot, _) in _sessions.items()}


def _md5(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def _get_cached(key: str) -> dict | None:
    entry = _cache.get(key)
    if entry:
        result, ts = entry
        if time.time() - ts < _CACHE_TTL:
            return result
        del _cache[key]
    return None


def _gc_cache(cache: dict, ttl: float = _CACHE_TTL) -> None:
    """공용 — 만료된 캐시 항목 제거 (TTL 초과)"""
    now = time.time()
    for k in [k for k, (_, ts) in list(cache.items()) if now - ts >= ttl]:
        cache.pop(k, None)


def _set_cache(key: str, result: dict):
    _cache[key] = (result, time.time())
    _gc_cache(_cache)


async def _get_or_infer(image_bytes: bytes) -> dict:
    """
    캐시 확인 → 추론 중이면 대기 → 없으면 추론 후 캐시 저장.
    같은 프레임으로 /detect, /segment가 동시에 와도 추론은 1번만 실행.
    """
    key = _md5(image_bytes)

    # 1) 캐시 히트
    cached = _get_cached(key)
    if cached is not None:
        return cached

    # 2) 추론 진행 중이면 끝날 때까지 대기
    if key in _pending:
        await _pending[key].wait()
        return _get_cached(key) or {}

    # 3) 직접 추론
    event = asyncio.Event()
    _pending[key] = event
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            shared_executor, inference_service.analyze_frame, image_bytes
        )
        _set_cache(key, result)
        return result
    finally:
        event.set()
        _pending.pop(key, None)


# ── 노면 클래스 → SafeStep status 매핑 ────────────────────────────────────
_CLS_TO_SAFESTEP = {
    "roadway_normal":              "road",
    "roadway_crosswalk":           "crosswalk",
    "alley_normal":                "alley",
    "alley_crosswalk":             "crosswalk",
    "alley_damaged":               "alley",
    "alley_speed_bump":            "alley",
    "sidewalk_asphalt":            "sidewalk",
    "sidewalk_blocks":             "sidewalk",
    "sidewalk_cement":             "sidewalk",
    "sidewalk_urethane":           "sidewalk",
    "sidewalk_soil_stone":         "sidewalk",
    "sidewalk_other":              "sidewalk",
    "sidewalk_damaged":            "sidewalk",
    "braille_guide_blocks_normal": "sidewalk",
    "braille_guide_blocks_damaged":"sidewalk",
    "caution_zone_grating":        "sidewalk",
    "caution_zone_manhole":        "sidewalk",
    "caution_zone_repair_zone":    "sidewalk",
    "caution_zone_stairs":         "sidewalk",
    "caution_zone_tree_zone":      "sidewalk",
    "bike_lane":                   "road",
}

# SURFACE_CLASS_TO_GROUP의 group → SafeStep status (정면 구역 status 결정용)
_GROUP_TO_STATUS = {
    "danger":  "road",
    "caution": "alley",
    "safe":    "sidewalk",
    "braille": "sidewalk",
    "bike":    "road",
}


# ══════════════════════════════════════════════════════════════════════════════
# POST /detect
# ══════════════════════════════════════════════════════════════════════════════
@router.post("/detect")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
):
    """
    SafeStep RemoteDetector.kt 호환 엔드포인트.

    반환 포맷:
    {
        "detections": [...],
        "message": "...",
        "dodge": "왼쪽|정면|오른쪽",
        "analysis": {...},
        "detail": {...}
    }

    헤더:
    - X-Client-Id: 클라이언트 식별자 (없으면 IP로 fallback)
    응답 코드:
    - 200: 정상
    - 429: 동시 접속 한도(MAX_CLIENTS) 초과
    """
    # 슬롯 체크 (X-Client-Id 없으면 IP fallback)
    client_id = x_client_id or (request.client.host if request.client else "unknown")
    slot = _check_slot(client_id)

    image_bytes = await file.read()
    update_latest_frame(image_bytes, slot=slot)   # 슬롯별 라이브 카메라 업데이트
    result = await _get_or_infer(image_bytes)

    detections = [
        {
            "label":      item["cls_name"],
            "label_ko":   item["group"],
            "confidence": round(float(item["conf"]), 3),
            "box":        item["bbox"],          # [x1, y1, x2, y2]
            "direction":  item.get("direction", "정면"),
            "depth_m":    item.get("depth_m"),
        }
        for item in result.get("obstacle_items", [])
    ]

    # message: Ollama ON이면 자연어 메시지, OFF면 규칙 기반 메시지
    message = result.get("message", "")

    # dodge: 회피 방향 (왼쪽/정면/오른쪽) - 진동 방향에 사용
    dodge_text = result.get("detail", {}).get("dodge", "")
    if "왼쪽" in dodge_text:
        dodge = "왼쪽"
    elif "오른쪽" in dodge_text:
        dodge = "오른쪽"
    else:
        dodge = "정면"

    logger.info(f"[SafeStep /detect] 탐지 {len(detections)}개 | dodge={dodge} | message={message}")

    # 대시보드 사이드바용 analysis 객체 (노면/장애물 카드 업데이트용)
    analysis = {
        "obstacle": result.get("obstacle", {}),
        "surface":  result.get("surface",  {}),
    }

    # 대시보드에 라이브 오버레이 브로드캐스트 (slot 정보 포함 → 해당 슬롯 뷰만 갱신)
    if dashboard_manager.count > 0:
        asyncio.create_task(dashboard_manager.broadcast({
            "type": "frame_overlay",
            "slot": slot,
            "surface_masks": result.get("surface_masks", {}),
            "obstacle_items": result.get("obstacle_items", []),
            "img_width": result.get("img_width", 0),
            "img_height": result.get("img_height", 0),
            "message": message,
            "detail": result.get("detail", {}),
            "analysis": {
                "obstacle": result.get("obstacle", {}),
                "surface": result.get("surface", {}),
            },
        }))

    # ── 학습 데이터 캡처 (ON일 때만, 5초/슬롯 간격) ──
    # to_thread로 백그라운드 실행 → 디스크 I/O가 이벤트 루프 블록하지 않음
    # (특히 5GB 한도 도달 시 prune이 수 초 걸려도 다른 슬롯 요청 영향 X)
    if capture_service.is_enabled():
        asyncio.create_task(
            asyncio.to_thread(
                capture_service.save, image_bytes, result, slot, client_id
            )
        )

    return {
        "detections": detections,
        "message":    message,
        "dodge":      dodge,
        "analysis":   analysis,
        "detail":     result.get("detail", {}),
        "depth_status": result.get("depth_status", {"enabled": True, "auto_off": False}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# POST /segment
# ══════════════════════════════════════════════════════════════════════════════
@router.post("/segment")
async def segment(
    request: Request,
    file: UploadFile = File(...),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
    include_mask: int = 0,   # 1이면 합쳐진 마스크 PNG(base64) 추가 — 대시보드 영상/사진 탭용
):
    """
    SafeStep SegmentationClient.kt 호환 엔드포인트.

    반환 포맷:
    {
        "status":               "sidewalk" | "road" | "crosswalk" | "alley" | "unknown",
        "ratios":               {"road": 0.1, "sidewalk": 0.8, "crosswalk": 0.0, "alley": 0.1},
        "front_cls":            "<세부 클래스명>",
        "left_status":          "<왼쪽 구역 status>",
        "right_status":         "<오른쪽 구역 status>",
        "traffic_light_color":  "red" | "green" | "",
        "mask_b64":             "<base64 PNG>"   # include_mask=1일 때만
    }
    """
    client_id = x_client_id or (request.client.host if request.client else "unknown")
    _check_slot(client_id)   # 슬롯 갱신만 (segment 결과는 슬롯 무관)

    image_bytes = await file.read()
    key = _md5(image_bytes)

    # 세그먼트 후처리 결과 캐시 (이미지 동일하면 재사용)
    # include_mask 여부도 캐시 키에 포함 (있는 응답 vs 없는 응답)
    cache_key = (key, bool(include_mask))
    seg_entry = _seg_cache.get(cache_key)
    if seg_entry:
        seg_result, ts = seg_entry
        if time.time() - ts < _CACHE_TTL:
            return seg_result
        del _seg_cache[cache_key]

    result = await _get_or_infer(image_bytes)
    surface = result.get("surface", {})

    # ── 비율 계산 (이미 계산된 class_ratios 사용 — PNG 디코드 안 함) ──
    class_ratios = surface.get("class_ratios", {})  # 0~100 (%)
    ratios = {"road": 0.0, "sidewalk": 0.0, "crosswalk": 0.0, "alley": 0.0}
    for cls_name, pct in class_ratios.items():
        key_ss = _CLS_TO_SAFESTEP.get(cls_name)
        if key_ss in ratios:
            ratios[key_ss] += pct / 100.0   # 0~1 비율로 환산

    # ── status 결정 (정면 구역 기준) ───────────────────────────────────
    front_zone = surface.get("정면") or {}
    front_cls = front_zone.get("cls_name", "")
    if "crosswalk" in front_cls:
        status = "crosswalk"
    else:
        status = _GROUP_TO_STATUS.get(front_zone.get("group", ""), "unknown")

    # ── 왼쪽 / 오른쪽 구역 status (방향 유도용) ────────────────────────
    def _zone_to_status(zone_info) -> str:
        if not zone_info:
            return "unknown"
        cls = zone_info.get("cls_name", "")
        if "crosswalk" in cls:
            return "crosswalk"
        return _GROUP_TO_STATUS.get(zone_info.get("group", ""), "unknown")

    left_status  = _zone_to_status(surface.get("왼쪽"))
    right_status = _zone_to_status(surface.get("오른쪽"))

    # ── 신호등 색상 (횡단보도일 때만) ────────────────────────────────────
    traffic_light_color = ""
    if status == "crosswalk":
        for item in result.get("obstacle_items", []):
            if item.get("cls_name") == "traffic_light" and item.get("light_color"):
                traffic_light_color = item["light_color"]   # "red" | "green"
                break

    seg_result = {
        "status":              status,
        "ratios":              {k: round(v, 4) for k, v in ratios.items()},
        "front_cls":           front_cls,
        "left_status":         left_status,
        "right_status":        right_status,
        "traffic_light_color": traffic_light_color,
    }

    # ── (옵션) 합쳐진 마스크 PNG — 대시보드 영상/사진 탭이 ?include_mask=1로 요청 ──
    if include_mask:
        import base64 as _b64
        import cv2 as _cv2
        import numpy as _np
        surface_masks = result.get("surface_masks", {})
        img_w = result.get("img_width", 640)
        img_h = result.get("img_height", 480)
        combined = _np.zeros((img_h, img_w, 4), dtype=_np.uint8)
        for cls_name, b64_png in surface_masks.items():
            b64_data = b64_png.split(",", 1)[1] if "," in b64_png else b64_png
            try:
                png_bytes = _b64.b64decode(b64_data)
                arr = _np.frombuffer(png_bytes, dtype=_np.uint8)
                mask_img = _cv2.imdecode(arr, _cv2.IMREAD_UNCHANGED)
            except Exception:
                continue
            if mask_img is None or mask_img.ndim < 3:
                continue
            alpha = mask_img[:, :, 3]
            mask = alpha > 0
            combined[mask] = mask_img[mask]
        ok, buf = _cv2.imencode(".png", combined, [_cv2.IMWRITE_PNG_COMPRESSION, 6])
        if ok:
            seg_result["mask_b64"] = _b64.b64encode(buf.tobytes()).decode()

    _seg_cache[cache_key] = (seg_result, time.time())
    _gc_cache(_seg_cache)

    logger.info(f"[SafeStep /segment] status={status}, front={front_cls}, mask={'on' if include_mask else 'off'}")
    return seg_result


# ══════════════════════════════════════════════════════════════════════════════
# 학습 데이터 캡처 토글 / 상태
# ══════════════════════════════════════════════════════════════════════════════
@router.post("/capture/toggle")
async def capture_toggle(enable: bool | None = None):
    """
    캡처 ON/OFF 토글.
    - enable 파라미터 명시 시 그 값으로 설정
    - 미지정 시 현재 상태의 반대로 설정
    """
    cur = capture_service.is_enabled()
    target = (not cur) if enable is None else enable
    capture_service.set_enabled(target)
    return capture_service.get_status()


@router.get("/capture/status")
async def capture_status():
    """대시보드 폴링용 — 현재 ON/OFF, 누적 장수, 용량 등"""
    return capture_service.get_status()

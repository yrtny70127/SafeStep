"""
학습 데이터 자동 캡처 서비스.

기능:
- 5초 간격(슬롯별)으로 원본 이미지(.jpg) + 메타데이터(.json) + 노면 마스크(.png) 저장
- 디스크 한도(5GB) 초과 시 가장 오래된 날짜 폴더부터 자동 삭제
- 한도 도달이 반복되면 자동 OFF (사용자 알림용 플래그 노출)
- 기본 OFF, 대시보드에서 토글

저장 경로:
    data/captures/<YYYY-MM-DD>/<HH-MM-SS>_slot<N>_<md5앞8>_<idx>.jpg
                                                              .json
                                                              _mask.png
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from backend.utils.logger import get_logger
from backend.services.inference_service import SURFACE_CLASS_TO_GROUP

logger = get_logger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────
CAPTURE_ROOT      = Path("data/captures")
SAVE_INTERVAL_SEC = 5.0                 # 슬롯별 저장 간격
DISK_LIMIT_BYTES  = 5 * 1024 ** 3       # 5GB
PRUNE_TARGET_PCT  = 0.85                # 정리 후 이 비율 이하로 떨어뜨림

# 노면 클래스명 → 학습용 마스크 픽셀 ID (1부터, 0은 background)
_SURFACE_CLS_LIST = sorted(SURFACE_CLASS_TO_GROUP.keys())
SURFACE_CLASS_ID  = {name: i + 1 for i, name in enumerate(_SURFACE_CLS_LIST)}
SURFACE_ID_TO_CLS = {v: k for k, v in SURFACE_CLASS_ID.items()}


# ── 상태 ──────────────────────────────────────────────────────────────
_enabled = False
_total_count = 0
_total_bytes = 0
_last_save_ts: dict[int, float] = {}    # {slot: last_ts}
_frame_idx = 0
_lock = threading.Lock()                # 동시 저장 직렬화 (디스크 보호)
_disk_full_disabled = False             # 디스크 한도로 자동 OFF 됐는지


def _scan_total_bytes() -> int:
    """captures 폴더 전체 크기 (재귀)"""
    if not CAPTURE_ROOT.exists():
        return 0
    total = 0
    for dirpath, _, files in os.walk(CAPTURE_ROOT):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _initialize_state():
    """앱 시작 시 폴더 스캔하여 누적 통계 복원"""
    global _total_bytes, _total_count
    if not CAPTURE_ROOT.exists():
        return
    _total_bytes = _scan_total_bytes()
    _total_count = sum(1 for p in CAPTURE_ROOT.rglob("*.jpg"))
    logger.info(f"[Capture] 기존 데이터 {_total_count}장, "
                f"{_total_bytes / 1024**2:.1f}MB 발견")


def _prune_oldest_to(target_bytes: int) -> int:
    """가장 오래된 날짜 폴더부터 삭제하여 target_bytes 이하로. 삭제된 바이트 수 반환."""
    if not CAPTURE_ROOT.exists():
        return 0
    date_dirs = sorted(d for d in CAPTURE_ROOT.iterdir() if d.is_dir())
    freed = 0
    cur = _scan_total_bytes()
    for d in date_dirs:
        if cur <= target_bytes:
            break
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        try:
            shutil.rmtree(d)
            freed += size
            cur -= size
            logger.warning(f"[Capture] 디스크 한도 초과 → {d.name} 삭제 ({size/1024**2:.1f}MB)")
        except OSError as e:
            logger.error(f"[Capture] 폴더 삭제 실패 {d}: {e}")
    return freed


# ── 공개 API ──────────────────────────────────────────────────────────
def set_enabled(value: bool) -> bool:
    """캡처 ON/OFF 토글. 디스크 자동 OFF 상태였으면 사용자 ON으로 해제"""
    global _enabled, _disk_full_disabled
    _enabled = value
    if value:
        _disk_full_disabled = False
        CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
        logger.info("[Capture] ON")
    else:
        logger.info("[Capture] OFF")
    return _enabled


def is_enabled() -> bool:
    return _enabled and not _disk_full_disabled


def get_status() -> dict:
    return {
        "enabled":           _enabled,
        "auto_disabled":     _disk_full_disabled,
        "total_count":       _total_count,
        "total_bytes":       _total_bytes,
        "total_mb":          round(_total_bytes / 1024**2, 1),
        "limit_mb":          round(DISK_LIMIT_BYTES / 1024**2, 1),
        "interval_sec":      SAVE_INTERVAL_SEC,
        "folder":            str(CAPTURE_ROOT.resolve()),
    }


def _should_save(slot: int) -> bool:
    """슬롯별 5초 간격 체크"""
    now = time.time()
    last = _last_save_ts.get(slot, 0)
    if now - last < SAVE_INTERVAL_SEC:
        return False
    _last_save_ts[slot] = now
    return True


def _build_surface_mask(surface_masks_b64: dict) -> np.ndarray | None:
    """
    surface_masks (cls_name → BGRA base64 PNG) 를
    단일 채널 클래스 ID 마스크(uint8)로 합치기. 학습용 GT.
    """
    if not surface_masks_b64:
        return None
    first_arr = None
    for cls_name, b64_png in surface_masks_b64.items():
        b64_data = b64_png.split(",", 1)[1] if "," in b64_png else b64_png
        try:
            png_bytes = base64.b64decode(b64_data)
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            mask_img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception:
            continue
        if mask_img is None or mask_img.ndim < 3:
            continue
        h, w = mask_img.shape[:2]
        if first_arr is None:
            first_arr = np.zeros((h, w), dtype=np.uint8)
        cls_id = SURFACE_CLASS_ID.get(cls_name)
        if cls_id is None:
            continue
        alpha = mask_img[:, :, 3]
        first_arr[alpha > 0] = cls_id
    return first_arr


def save(image_bytes: bytes, result: dict, slot: int, client_id: str = "") -> bool:
    """
    1프레임 저장. result는 inference_service.analyze_frame() 결과.
    저장 시 True, 스킵/실패 시 False.
    """
    global _total_count, _total_bytes, _frame_idx, _disk_full_disabled

    if not is_enabled():
        return False
    if not _should_save(slot):
        return False

    try:
        # ── 디스크 한도 체크 ──
        with _lock:
            if _total_bytes > DISK_LIMIT_BYTES:
                target = int(DISK_LIMIT_BYTES * PRUNE_TARGET_PCT)
                freed = _prune_oldest_to(target)
                _total_bytes -= freed
                if _total_bytes > DISK_LIMIT_BYTES:
                    # 정리 후에도 한도 초과 → 자동 OFF
                    _disk_full_disabled = True
                    logger.error("[Capture] 디스크 정리 실패 → 자동 OFF")
                    return False

        # ── 파일명 생성 ──
        now = datetime.now()
        date_dir = CAPTURE_ROOT / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        import hashlib
        md5_short = hashlib.md5(image_bytes).hexdigest()[:8]
        # _frame_idx 증가는 lock으로 보호 (두 슬롯 동시 호출 시 race 방지)
        with _lock:
            _frame_idx += 1
            idx_now = _frame_idx
        base = f"{now.strftime('%H-%M-%S')}_slot{slot}_{md5_short}_{idx_now:06d}"

        jpg_path  = date_dir / f"{base}.jpg"
        json_path = date_dir / f"{base}.json"
        mask_path = date_dir / f"{base}_mask.png"

        # ── 1) 원본 JPEG ──
        jpg_path.write_bytes(image_bytes)
        size = len(image_bytes)

        # ── 2) 메타데이터 JSON (학습용 의사 라벨) ──
        meta = {
            "ts":            now.isoformat(timespec="milliseconds"),
            "client_id":     client_id,
            "slot":          slot,
            "md5":           md5_short,
            "img_width":     result.get("img_width"),
            "img_height":    result.get("img_height"),
            "obstacles": [
                {
                    "cls_name":   it.get("cls_name"),
                    "group":      it.get("group"),
                    "conf":       round(float(it.get("conf", 0.0)), 4),
                    "bbox":       it.get("bbox"),
                    "depth_m":    it.get("depth_m"),
                    "direction":  it.get("direction"),
                    "light_color": it.get("light_color"),
                }
                for it in result.get("obstacle_items", [])
            ],
            "surface_zones": result.get("surface", {}).get("zones", {}) if isinstance(result.get("surface"), dict) else {},
            "surface_class_ratios": result.get("surface", {}).get("class_ratios", {}) if isinstance(result.get("surface"), dict) else {},
            "depth_status": result.get("depth_status", {}),
        }
        json_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        json_path.write_bytes(json_bytes)
        size += len(json_bytes)

        # ── 3) 노면 학습용 마스크 PNG (픽셀 = 클래스 ID) ──
        train_mask = _build_surface_mask(result.get("surface_masks", {}))
        if train_mask is not None:
            ok, png_buf = cv2.imencode(".png", train_mask)
            if ok:
                mask_path.write_bytes(png_buf.tobytes())
                size += len(png_buf)

        with _lock:
            _total_count += 1
            _total_bytes += size
        return True

    except Exception as e:
        logger.error(f"[Capture] 저장 실패: {e}")
        return False


# ── 모듈 import 시 통계 복원 ──
_initialize_state()

"""
사진/영상 파일 업로드 → AI 추론 → 대시보드 브로드캐스트
POST /analyze-file
"""
import asyncio
import base64
import tempfile
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.routers.camera_router import dashboard_manager
from backend.services import inference_service
from backend.services.inference_service import shared_executor
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
_current_task: asyncio.Task | None = None  # 현재 분석 중인 태스크


@router.post("/ollama-toggle")
async def ollama_toggle():
    new_state = not inference_service.get_ollama_enabled()
    inference_service.set_ollama_enabled(new_state)
    logger.info(f"Ollama {'활성화' if new_state else '비활성화'}")
    return {"ollama_enabled": new_state}


@router.post("/depth-toggle")
async def depth_toggle():
    new_state = not inference_service.get_depth_enabled()
    inference_service.set_depth_enabled(new_state)
    logger.info(f"Depth {'활성화' if new_state else '비활성화'}")
    return {"depth_enabled": new_state}


@router.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    global _current_task

    # 이전 분석 태스크가 실행 중이면 취소 후 완전히 종료될 때까지 대기
    if _current_task and not _current_task.done():
        _current_task.cancel()
        try:
            await _current_task  # 완전히 멈출 때까지 기다림
        except asyncio.CancelledError:
            pass
        logger.info("이전 분석 취소 → 새 파일로 교체")
        await dashboard_manager.broadcast({"type": "video_done", "total_steps": 0, "cancelled": True})

    ct = file.content_type or ""
    if ct.startswith("image/"):
        contents = await file.read()
        _current_task = asyncio.create_task(_process_image(contents, file.filename))
        return {"status": "started", "type": "image", "filename": file.filename}
    elif ct.startswith("video/"):
        contents = await file.read()
        _current_task = asyncio.create_task(_process_video(contents, file.filename))
        return {"status": "started", "type": "video", "filename": file.filename}
    else:
        raise HTTPException(status_code=400, detail="사진 또는 동영상 파일만 업로드 가능합니다")


# ── 사진 처리 ──────────────────────────────────
async def _process_image(contents: bytes, filename: str):
    loop = asyncio.get_running_loop()

    arr = __import__('numpy').frombuffer(contents, dtype=__import__('numpy').uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        await dashboard_manager.broadcast({"type": "video_error", "message": "사진 파일을 열 수 없습니다"})
        await dashboard_manager.broadcast({"type": "video_done", "total_steps": 0})
        return

    await dashboard_manager.broadcast({"type": "video_start", "filename": filename, "total_steps": 1})

    # 전송용 리사이즈 (최대 960px, 메시지 크기 축소)
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv2.resize(frame, (960, int(h * 960 / w)))

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    size_kb = len(buf) / 1024

    try:
        result = await loop.run_in_executor(shared_executor, inference_service.analyze_frame, buf.tobytes())
    except Exception as e:
        logger.error(f"사진 추론 오류: {e}")
        result = {"message": f"추론 오류: {e}", "obstacle": {}, "surface": {}}

    await dashboard_manager.broadcast({
        "type": "frame", "source": "video", "frame_id": 1,
        "base_image": result.get("base_image", img_b64),
        "surface_masks": result.get("surface_masks", {}),
        "obstacle_items": result.get("obstacle_items", []),
        "size_kb": round(size_kb, 1),
        "timestamp": int(time.time() * 1000),
        "message": result.get("message", ""),
        "detail": result.get("detail", {}),
        "step": 1, "total_steps": 1,
        "analysis": {"obstacle": result.get("obstacle", {}), "surface": result.get("surface", {})},
    })
    await dashboard_manager.broadcast({"type": "video_done", "total_steps": 1})
    logger.info(f"사진 분석 완료: {filename}")


# ── 영상 처리 (전체 프레임 순차 분석) ──────────
async def _process_video(contents: bytes, filename: str):
    loop = asyncio.get_running_loop()

    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as f:
        f.write(contents)
        tmp_path = f.name

    cap = None
    step = 0
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            await dashboard_manager.broadcast({"type": "video_error", "message": "영상 파일을 열 수 없습니다"})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"영상 분석 시작: {filename} ({total_frames}프레임)")

        await dashboard_manager.broadcast({
            "type": "video_start", "filename": filename, "total_steps": total_frames,
        })

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            step += 1

            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960 / w)))

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            size_kb = len(buf) / 1024

            try:
                result = await loop.run_in_executor(shared_executor, inference_service.analyze_frame, buf.tobytes())
            except Exception as e:
                logger.error(f"추론 오류 (프레임 {step}): {e}")
                result = {"message": "추론 오류", "obstacle": {}, "surface": {}}

            await dashboard_manager.broadcast({
                "type": "frame", "source": "video", "frame_id": step,
                "base_image": result.get("base_image", img_b64),
                "surface_masks": result.get("surface_masks", {}),
                "obstacle_items": result.get("obstacle_items", []),
                "size_kb": round(size_kb, 1),
                "timestamp": int(time.time() * 1000),
                "message": result.get("message", ""),
                "detail": result.get("detail", {}),
                "step": step, "total_steps": total_frames,
                "analysis": {"obstacle": result.get("obstacle", {}), "surface": result.get("surface", {})},
            })
            await asyncio.sleep(0)  # 이벤트 루프에 양보

        logger.info(f"영상 분석 완료: {filename} ({step}프레임)")

    except asyncio.CancelledError:
        logger.info(f"영상 분석 취소됨: {filename} ({step}프레임 완료)")
    except Exception as e:
        logger.exception(f"영상 분석 오류: {e}")
        await dashboard_manager.broadcast({"type": "video_error", "message": f"분석 오류: {str(e)}"})
    finally:
        if cap is not None:
            cap.release()
        Path(tmp_path).unlink(missing_ok=True)

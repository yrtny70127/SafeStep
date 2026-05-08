"""
VisionGuide - 시각장애인을 위한 실시간 AI 보행 비서
FastAPI 메인 진입점

실행 방법 (프로젝트 루트에서):
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

사전 준비:
    - .env 파일에 NGROK_AUTH_TOKEN, NGROK_DOMAIN 설정
    - tools/ngrok.exe 배치
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.routers import camera_router, video_router, nav_router, safestep_router
from backend.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# ============================================
# FastAPI 앱 초기화
# ============================================
app = FastAPI(
    title="VisionGuide",
    description="시각장애인을 위한 실시간 AI 보행 비서",
    version="0.1.0",
)

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 라우터 등록
# ============================================
app.include_router(camera_router.router, tags=["camera"])
app.include_router(video_router.router, tags=["video"])
app.include_router(nav_router.router)
app.include_router(safestep_router.router, tags=["safestep"])

# ============================================
# 프론트엔드 정적 파일 서빙
# ============================================
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    """핸드폰용 메인 페이지"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/dashboard")
async def dashboard():
    """노트북용 실시간 대시보드"""
    return FileResponse(FRONTEND_DIR / "dashboard.html",
                        headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "VisionGuide"}


@app.on_event("startup")
async def on_startup():
    import asyncio
    from backend.services.inference_service import _load_models, shared_executor

    logger.info("=" * 50)
    logger.info("🦮 VisionGuide 서버 시작 준비 중...")
    logger.info("=" * 50)

    # AI 모델 사전 로드 (완료될 때까지 대기)
    logger.info("🤖 AI 모델 로딩 중...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(shared_executor, _load_models)
    logger.info("✅ AI 모델 로딩 완료")

    # ngrok 자동 실행
    try:
        from pyngrok import ngrok, conf
        ngrok_exe = Path(__file__).parent.parent / "tools" / "ngrok.exe"
        if ngrok_exe.exists():
            conf.get_default().ngrok_path = str(ngrok_exe)

        auth_token = os.getenv("NGROK_AUTH_TOKEN", "")
        domain = os.getenv("NGROK_DOMAIN", "")

        if auth_token and auth_token != "여기에_토큰_입력":
            ngrok.set_auth_token(auth_token)
            if domain and domain != "여기에_고정도메인_입력":
                tunnel = ngrok.connect(8000, domain=domain)
            else:
                tunnel = ngrok.connect(8000)
            phone_url = tunnel.public_url
        else:
            phone_url = "ngrok URL (HTTPS 필수) - .env 설정 필요"
    except ImportError:
        phone_url = "ngrok URL (HTTPS 필수) - pyngrok 미설치"
    except Exception as e:
        logger.warning(f"ngrok 실행 실패: {e}")
        phone_url = "ngrok 실행 실패"

    logger.info("=" * 50)
    logger.info("🦮 VisionGuide 서버 시작")
    logger.info("=" * 50)
    logger.info(f"📱 핸드폰 접속: {phone_url}")
    logger.info("💻 노트북 대시보드: http://localhost:8000/dashboard")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("👋 VisionGuide 서버 종료")

# SafeStep 음성 어시스턴트 — /assistant 엔드포인트 (LangChain + Gemini)
#
# langchain-kr 튜토리얼 패턴을 따라 LangChain 으로 구현했습니다.
# 참조: https://github.com/teddylee777/langchain-kr
#
# 핵심 흐름:
#   사용자 음성 텍스트 + 화면 컨텍스트 → Gemini bind_tools → action JSON
#
# 화면별 동작:
#   - SplashActivity (모드 선택 화면): 카메라 모드 / 동영상 테스트 진입만
#   - MapActivity   (지도 화면)        : 내비·위치·주변 설명 등 풀 기능
#
# 통합 방법:
#   1. 이 파일을 SafeStep/server/ 폴더에 복사
#   2. server.py 에 라우터 등록 (INTEGRATION.md 참고)
#   3. .env 에 GOOGLE_API_KEY 추가
#   4. pip install langchain langchain-google-genai python-dotenv

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API KEY 정보로드 (langchain-kr 표준 패턴)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 설정 — Gemini 2.0 Flash (무료 티어 / 빠름)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_NAME  = "gemini-2.0-flash"
TEMPERATURE = 0.0   # 분류 작업이므로 결정적 출력

llm = None
if os.getenv("GOOGLE_API_KEY"):
    try:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE)
        print(f"[Assistant] LangChain Gemini({MODEL_NAME}) 로드 완료 ✅")
    except Exception as e:
        print(f"[Assistant] LLM 초기화 실패: {e}")
else:
    print("[Assistant] ⚠️  GOOGLE_API_KEY 미설정 — /assistant 비활성화")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool 정의 — @tool 데코레이터 (langchain-kr 표준 패턴)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 주의: 아래 도구 함수들은 서버에서 직접 실행되지 않습니다.
# LangChain 은 함수 시그니처와 docstring 을 LLM 에 전달해서
# "어떤 도구를 호출할지" 분류만 시킵니다.
# 실제 동작은 안드로이드 앱(Kotlin)이 action 이름을 보고 실행합니다.

# ─────────────────────────────────────────────
# SplashActivity (모드 선택 화면) 전용 도구
# ─────────────────────────────────────────────

@tool
def enter_camera_mode() -> str:
    """SafeStep의 메인 화면(카메라 모드)으로 진입할 때 호출하세요.
    예: "카메라 모드", "시작", "카메라로 들어가", "보행 모드", "출발"
    """
    return "카메라 모드를 시작합니다."


@tool
def enter_video_test_mode() -> str:
    """동영상 테스트 모드로 진입할 때 호출하세요.
    예: "동영상 테스트", "비디오 테스트", "테스트 모드"
    """
    return "동영상 테스트 모드를 시작합니다."


# ─────────────────────────────────────────────
# MapActivity (지도/카메라 화면) 전용 도구
# ─────────────────────────────────────────────

@tool
def start_navigation(destination: str) -> str:
    """사용자가 특정 목적지로 도보 내비게이션을 시작하고 싶을 때 호출하세요.

    예: "강남역 가줘", "스타벅스로 안내해줘", "집까지 길 알려줘"

    Args:
        destination: 목적지 이름 (예: 강남역, 스타벅스, 우리집)
    """
    return f"{destination}으로 안내를 시작합니다."


@tool
def cancel_navigation() -> str:
    """현재 진행 중인 내비게이션을 취소하거나 종료할 때 호출하세요.
    예: "내비 그만", "안내 종료", "취소해"
    """
    return "내비게이션을 종료합니다."


@tool
def reroute() -> str:
    """현재 목적지로 가는 경로를 다시 탐색하고 싶을 때 호출하세요.
    예: "다시 길 찾아줘", "재탐색", "경로 새로 찾아줘"
    """
    return "경로를 다시 탐색합니다."


@tool
def get_current_location() -> str:
    """사용자가 현재 자기 위치(주소·좌표)를 알고 싶어할 때 호출하세요.
    예: "지금 어디야?", "현재 위치 알려줘", "내 위치 어디?"
    """
    return "현재 위치를 안내합니다."


@tool
def get_remaining_info() -> str:
    """내비게이션 중 남은 거리 또는 도착 예정 시간을 알려달라고 할 때 호출하세요.
    내비게이션 중이 아닐 때는 호출하지 마세요.
    예: "얼마나 남았어?", "몇 분 걸려?", "남은 거리?"
    """
    return "남은 거리를 안내합니다."


@tool
def describe_surroundings() -> str:
    """사용자가 카메라에 보이는 주변 상황(객체·장애물)을 설명해달라고 할 때 호출하세요.
    예: "주변에 뭐 있어?", "앞에 뭐가 있어?", "주위 알려줘"
    """
    return "주변 상황을 안내합니다."


@tool
def find_nearby_poi(category: str) -> str:
    """주변의 특정 종류 시설(화장실·편의점·약국 등)을 Tmap POI 검색으로 찾을 때 호출하세요.
    "describe_surroundings"는 카메라에 보이는 객체 설명이고,
    이 도구는 지도상의 시설 검색입니다. 헷갈리지 마세요.

    예: "근처 화장실 찾아줘", "편의점 어디 있어?", "약국 알려줘"

    Args:
        category: 검색할 시설 종류 (예: 화장실, 편의점, 약국, 카페, 지하철역)
    """
    return f"주변 {category}을(를) 검색합니다."


# ─────────────────────────────────────────────
# 화면별 도구 세트
# ─────────────────────────────────────────────
SPLASH_TOOLS = [enter_camera_mode, enter_video_test_mode]
MAP_TOOLS = [
    start_navigation,
    cancel_navigation,
    reroute,
    get_current_location,
    get_remaining_info,
    describe_surroundings,
    find_nearby_poi,
]


def _tools_for_screen(screen: str):
    """현재 화면에 맞는 도구 리스트 반환."""
    if screen == "splash":
        return SPLASH_TOOLS
    return MAP_TOOLS   # 기본값: MapActivity


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 시스템 프롬프트 (화면별로 다르게)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM_PROMPT_BASE = """당신은 시각장애인용 보행 안전 앱 SafeStep의 음성 어시스턴트입니다.

[역할]
- 사용자 발화를 분석해 등록된 도구(tool) 중 가장 적절한 것을 호출하세요.
- 도구 호출이 필요 없는 일반 대화/질문이면 도구 호출 없이 한국어로 짧게 답하세요.
- 응답은 항상 짧고 명확해야 합니다. 시각장애인이 음성으로 듣기 때문입니다.
- 1~2문장 이내로 간결하게 답하세요.

[말투]
- 존댓말, 따뜻하고 차분하게.
- 불필요한 설명·반복 금지.
"""

SPLASH_GUIDE = """
[현재 화면: 모드 선택 화면]
- 사용자는 카메라 모드 또는 동영상 테스트 모드를 선택해야 합니다.
- 등록된 두 도구(enter_camera_mode, enter_video_test_mode)만 호출 가능합니다.
- 그 외 발화에는 "카메라 모드 또는 동영상 테스트 중 선택해주세요." 라고 답하세요.
"""

MAP_GUIDE = """
[현재 화면: 지도/카메라 화면]
- 사용자 메시지에는 현재 앱 상태(내비 진행 여부, 현재 위치, 최근 감지 객체 등)가
  함께 제공될 수 있습니다. 이를 참고해 더 정확하게 답하세요.
- 예: is_navigating=False 인데 "남은 거리?" 라고 물으면 get_remaining_info 를
  호출하지 말고 "현재 내비게이션 중이 아닙니다." 라고 답하세요.
- describe_surroundings(카메라에 보이는 객체)와 find_nearby_poi(지도상 시설)는
  서로 다른 도구입니다. 사용자 의도에 맞게 골라주세요.
"""


def _system_prompt_for(screen: str) -> str:
    if screen == "splash":
        return SYSTEM_PROMPT_BASE + SPLASH_GUIDE
    return SYSTEM_PROMPT_BASE + MAP_GUIDE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic 스키마
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AssistantRequest(BaseModel):
    text: str
    context: dict[str, Any] = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI 라우터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
router = APIRouter(tags=["assistant"])


@router.post("/assistant")
async def assistant(req: AssistantRequest):
    """음성 텍스트 → LangChain Gemini → action JSON 반환."""
    if llm is None:
        raise HTTPException(503, "LLM 이 초기화되지 않았습니다. GOOGLE_API_KEY 를 확인하세요.")
    if not req.text.strip():
        raise HTTPException(400, "text 가 비어있습니다.")

    screen = str(req.context.get("screen", "map"))
    tools  = _tools_for_screen(screen)

    # 화면에 맞는 시스템 프롬프트 + 도구를 LLM 에 결합
    llm_with_tools = llm.bind_tools(tools)

    user_text = _build_user_message(req.text, req.context)
    messages = [
        SystemMessage(content=_system_prompt_for(screen)),
        HumanMessage(content=user_text),
    ]

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        print(f"[Assistant] LLM 호출 실패: {e}")
        raise HTTPException(502, f"LLM 호출 실패: {e}")

    return JSONResponse(_parse_response(response))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 응답 파싱
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _parse_response(response) -> dict[str, Any]:
    """
    LangChain AIMessage 응답 → 앱이 쓰기 좋은 형태로 변환.

    response.tool_calls 예시:
        [{"name": "start_navigation",
          "args": {"destination": "강남역"},
          "id": "...", "type": "tool_call"}]
    """
    tool_calls = getattr(response, "tool_calls", None) or []
    text_content = (response.content or "").strip()

    if tool_calls:
        tc = tool_calls[0]
        action = tc.get("name", "speak_only")
        params = tc.get("args", {}) or {}
        tts = text_content or _default_tts_for(action, params)
        return {"action": action, "params": params, "tts": tts}

    # 도구 호출 없음 → 일반 응답
    return {
        "action": "speak_only",
        "params": {},
        "tts": text_content or "죄송합니다. 다시 말씀해주세요.",
    }


def _default_tts_for(action: str, params: dict[str, Any]) -> str:
    """LLM 이 tts 텍스트를 안 줬을 때의 기본 멘트."""
    if action == "enter_camera_mode":
        return "카메라 모드를 시작합니다."
    if action == "enter_video_test_mode":
        return "동영상 테스트 모드를 시작합니다."
    if action == "start_navigation":
        return f"{params.get('destination', '목적지')}으로 안내를 시작합니다."
    if action == "cancel_navigation":
        return "내비게이션을 종료합니다."
    if action == "reroute":
        return "경로를 다시 탐색합니다."
    if action == "find_nearby_poi":
        return f"주변 {params.get('category', '시설')}을(를) 검색합니다."
    return "네, 알겠습니다."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 사용자 메시지 빌더 — 컨텍스트 결합
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _build_user_message(text: str, context: dict[str, Any]) -> str:
    """사용자 발화 + 앱 상태 컨텍스트를 하나의 메시지로 조립."""
    if not context:
        return text

    lines = ["[현재 앱 상태]"]
    if context.get("screen"):
        lines.append(f"- 화면: {context['screen']}")
    if "is_navigating" in context:
        lines.append(f"- 내비 진행중: {bool(context['is_navigating'])}")
    if context.get("current_lat") is not None and context.get("current_lon") is not None:
        lines.append(f"- 현재 위치: ({context['current_lat']:.5f}, {context['current_lon']:.5f})")
    if context.get("remaining_distance_m") is not None:
        lines.append(f"- 남은 거리: {context['remaining_distance_m']}m")
    if context.get("recent_detections"):
        labels = ", ".join(context["recent_detections"][:5])
        lines.append(f"- 최근 감지된 객체: {labels}")

    lines.append("")
    lines.append("[사용자 발화]")
    lines.append(text)
    return "\n".join(lines)

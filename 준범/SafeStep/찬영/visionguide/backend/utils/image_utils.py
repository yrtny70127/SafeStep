"""
이미지 전처리 유틸리티
- 프론트엔드에서 보낸 base64 이미지를 디코딩
- 추후 OpenCV/PIL로 변환하기 좋은 형태로 처리
"""
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime


def decode_base64_image(data_url: str) -> bytes:
    """
    data URL (e.g. 'data:image/jpeg;base64,/9j/4AAQ...') 형식의 문자열을
    raw 바이트로 변환한다.
    """
    if "," in data_url:
        # 'data:image/jpeg;base64,<DATA>' → '<DATA>'만 추출
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    return base64.b64decode(encoded)


def save_frame_for_debug(image_bytes: bytes, save_dir: str = "debug_frames") -> Path:
    """
    디버깅용으로 받은 프레임을 디스크에 저장한다.
    Week 1 단계에서 '진짜로 이미지가 잘 들어왔는지' 눈으로 확인할 때 사용.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = save_path / f"frame_{timestamp}.jpg"

    file_path.write_bytes(image_bytes)
    return file_path

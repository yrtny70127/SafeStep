"""
VisionGuide - Surface Segmentation 웹캠 실시간 테스트

21개 클래스를 시각장애인 관점의 4개 그룹으로 재분류:
  🚶 SAFE      - 인도, 점자블록(정상)
  ⚠️  CAUTION   - 점자블록(파손), 주의구역(맨홀, 계단, 공사, 그레이팅, 가로수)
  🚧 DANGER    - 차도, 골목, 자전거 도로
  ❓ NEUTRAL   - 그 외

사용 예:
    python test_surface_seg.py --model path/to/best.pt
    python test_surface_seg.py --model best.pt --camera 1 --conf 0.4

조작:
    q / ESC : 종료
    s       : 현재 화면 캡처 저장 (captures/ 폴더)
    p       : 일시정지 / 재개
    g       : 그룹 색상 / 원본 색상 전환
    h       : 도움말 토글
"""
import argparse
import time
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "❌ ultralytics 패키지가 없습니다.\n"
        "   pip install ultralytics 로 설치하세요."
    )


# ============================================
# 1. 클래스 → 의미 그룹 매핑
# ============================================
# 그룹 키: (표시명, BGR 색상, 레벨)
#   레벨: 0=safe, 1=caution, 2=danger, 3=neutral
GROUP_META = {
    "safe":    ("SAFE",    (120, 220, 100), 0),   # 연두
    "caution": ("CAUTION", ( 60, 200, 255), 1),   # 주황
    "danger":  ("DANGER",  ( 60,  60, 240), 2),   # 빨강
    "bike":    ("BIKE",    (200, 120, 255), 2),   # 분홍 (위험)
    "braille": ("BRAILLE", ( 80, 240, 255), 0),   # 노랑 (안전)
    "neutral": ("OTHER",   (180, 180, 180), 3),   # 회색
}

#
# 📌 공식 AI-Hub Surface Masking 데이터셋 설명 기준으로 매핑
#    - roadway: "차만 다닐 수 있는 길"            → DANGER
#    - alley:   "사람과 차가 함께 다닐 수 있는 길" → CAUTION (공유 공간)
#
CLASS_TO_GROUP = {
    # 🚧 차도 (DANGER) - 차만 다님
    "roadway_normal":    "danger",
    "roadway_crosswalk": "danger",

    # ⚠️ 골목 (CAUTION) - 사람+차 공유 공간
    #    (공식: "사람과 차가 함께 다닐 수 있는 길")
    "alley_normal":      "caution",
    "alley_crosswalk":   "caution",
    "alley_damaged":     "caution",
    "alley_speed_bump":  "caution",

    # 🚶 인도 (SAFE)
    "sidewalk_asphalt":    "safe",
    "sidewalk_blocks":     "safe",
    "sidewalk_cement":     "safe",
    "sidewalk_urethane":   "safe",
    "sidewalk_soil_stone": "safe",
    "sidewalk_other":      "safe",
    "sidewalk_damaged":    "caution",   # 파손 인도는 주의

    # 🦯 점자블록
    "braille_guide_blocks_normal":   "braille",   # 안전 + 유도
    "braille_guide_blocks_damaged":  "caution",

    # ⚠️ 주의구역
    "caution_zone_grating":     "caution",   # 그레이팅 (배수로 덮개)
    "caution_zone_manhole":     "caution",   # 맨홀
    "caution_zone_repair_zone": "caution",   # 보수구역 (공사)
    "caution_zone_stairs":      "caution",   # 계단
    "caution_zone_tree_zone":   "caution",   # 가로수영역

    # 🚴 자전거 도로
    "bike_lane": "bike",
}

# 한국어 표시명 (콘솔 출력용, OpenCV 화면엔 영어로 표시)
KR_LABEL = {
    "roadway_normal":     "차도",
    "roadway_crosswalk":  "차도 횡단보도",

    "alley_normal":       "골목길",
    "alley_crosswalk":    "골목 횡단보도",
    "alley_damaged":      "파손 골목",
    "alley_speed_bump":   "과속방지턱",

    "sidewalk_asphalt":    "인도 (아스팔트)",
    "sidewalk_blocks":     "인도 (보도블럭)",
    "sidewalk_cement":     "인도 (시멘트)",
    "sidewalk_urethane":   "인도 (우레탄)",
    "sidewalk_soil_stone": "인도 (흙/돌)",
    "sidewalk_other":      "인도 (기타)",
    "sidewalk_damaged":    "파손 인도",

    "braille_guide_blocks_normal":   "점자블록",
    "braille_guide_blocks_damaged":  "파손 점자블록",

    "caution_zone_grating":     "그레이팅",
    "caution_zone_manhole":     "맨홀",
    "caution_zone_repair_zone": "공사 구역",
    "caution_zone_stairs":      "계단",
    "caution_zone_tree_zone":   "가로수 영역",

    "bike_lane": "자전거 도로",
}


# ============================================
# 2. 팔레트 (원본 모드용)
# ============================================
def make_palette(n_classes: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 179, n_classes, endpoint=False).astype(np.uint8)
    hsv = np.stack([hues,
                    np.full(n_classes, 220, dtype=np.uint8),
                    np.full(n_classes, 255, dtype=np.uint8)], axis=1)
    bgr = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    idx = rng.permutation(n_classes)
    return bgr[idx].tolist()


# ============================================
# 3. 파싱
# ============================================
def parse_args():
    parser = argparse.ArgumentParser(description="Surface Segmentation 웹캠 테스트")
    parser.add_argument("--model", "-m",
                        default="ai_models/yolo_surface/weights/best.pt",
                        help="학습된 YOLO-seg 모델 경로")
    parser.add_argument("--camera", "-c", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--conf", type=float, default=0.35, help="신뢰도 임계값")
    parser.add_argument("--imgsz", type=int, default=640, help="추론 이미지 크기")
    parser.add_argument("--width", type=int, default=1280, help="웹캠 가로")
    parser.add_argument("--height", type=int, default=720, help="웹캠 세로")
    return parser.parse_args()


# ============================================
# 4. 세그멘테이션 오버레이
# ============================================
def render_segmentation(frame, result, class_names, group_mode=True,
                        class_palette=None, alpha=0.5):
    """
    group_mode=True  : 의미 그룹별 색상 (safe=초록, danger=빨강 등)
    group_mode=False : 클래스별 고유 색상
    """
    if result.masks is None or len(result.masks) == 0:
        return frame, {}, {}

    h, w = frame.shape[:2]
    overlay = frame.copy()

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    class_pixels = defaultdict(int)
    group_pixels = defaultdict(int)
    total = h * w

    # 각 마스크 그리기
    for mask, cls_id in zip(masks, classes):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        bool_mask = mask_resized > 0.5

        cls_name = class_names.get(cls_id, f"cls{cls_id}")
        group = CLASS_TO_GROUP.get(cls_name, "neutral")

        if group_mode:
            color = GROUP_META[group][1]
        else:
            color = class_palette[cls_id % len(class_palette)]

        overlay[bool_mask] = color

        pixels = int(bool_mask.sum())
        class_pixels[cls_id] += pixels
        group_pixels[group] += pixels

    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 외곽선 + 라벨
    for mask, cls_id in zip(masks, classes):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        bool_mask = (mask_resized > 0.5).astype(np.uint8)

        cls_name = class_names.get(cls_id, f"cls{cls_id}")
        group = CLASS_TO_GROUP.get(cls_name, "neutral")

        if group_mode:
            color = GROUP_META[group][1]
        else:
            color = class_palette[cls_id % len(class_palette)]

        contours, _ = cv2.findContours(bool_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, color, 2)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:  # 너무 작은 건 라벨 생략
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = cls_name.replace("_", " ")
                    draw_label(blended, label, (cx, cy), color)

    ratios = {c: (p / total) * 100 for c, p in class_pixels.items()}
    group_ratios = {g: (p / total) * 100 for g, p in group_pixels.items()}
    return blended, ratios, group_ratios


def draw_label(img, text, center, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = center
    x1, y1 = x - tw // 2 - 4, y - th // 2 - 4
    x2, y2 = x + tw // 2 + 4, y + th // 2 + 4
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    cv2.putText(img, text, (x - tw // 2, y + th // 2 + 1),
                font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


# ============================================
# 5. 발 앞 영역 분석 (시각장애인 안내 핵심)
# ============================================
def analyze_foot_zone(result, class_names, frame_shape):
    """
    화면 하단 중앙 (발 앞) 영역에 어떤 클래스가 있는지 분석.
    → 시각장애인에게 '지금 발 앞이 어디인지' 안내할 때 사용.
    """
    if result.masks is None or len(result.masks) == 0:
        return None, {}

    h, w = frame_shape[:2]
    # 하단 1/3 + 좌우 중앙 1/2 영역 = 발 앞
    foot_y1 = int(h * 0.66)
    foot_y2 = h
    foot_x1 = int(w * 0.25)
    foot_x2 = int(w * 0.75)
    foot_area = (foot_y2 - foot_y1) * (foot_x2 - foot_x1)

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    foot_scores = defaultdict(int)
    for mask, cls_id in zip(masks, classes):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        foot_region = mask_resized[foot_y1:foot_y2, foot_x1:foot_x2]
        bool_mask = foot_region > 0.5
        foot_scores[cls_id] += int(bool_mask.sum())

    if not foot_scores:
        return None, {}

    # 가장 많이 차지한 클래스가 '발 앞'
    dominant_cls = max(foot_scores, key=foot_scores.get)
    dominant_pct = (foot_scores[dominant_cls] / foot_area) * 100

    dom_name = class_names.get(dominant_cls, f"cls{dominant_cls}")
    return (dom_name, dominant_pct), {
        class_names.get(c, f"cls{c}"): (s / foot_area) * 100
        for c, s in foot_scores.items()
    }


def generate_guidance(dominant, group_ratios):
    """발 앞 클래스 + 그룹 비율로 음성 안내문 생성 (Week 3 LLM 대체품)."""
    if dominant is None:
        return "탐지 없음", "neutral"

    cls_name, pct = dominant
    group = CLASS_TO_GROUP.get(cls_name, "neutral")
    kr = KR_LABEL.get(cls_name, cls_name)

    if group == "danger":
        msg = f"⚠️ 위험: 발 앞에 {kr}이(가) 있습니다"
    elif group == "bike":
        msg = f"⚠️ 자전거 도로 위입니다"
    elif group == "caution":
        msg = f"⚠️ 주의: {kr}"
    elif group == "braille":
        msg = f"✓ 점자블록 위에 있습니다"
    elif group == "safe":
        msg = f"✓ 인도 위 ({kr})"
    else:
        msg = f"현재 위치: {kr}"

    return msg, group


# ============================================
# 6. HUD 패널
# ============================================
def draw_hud(frame, fps, infer_ms, group_ratios, foot_info,
             paused=False, help_on=True, group_mode=True):
    h, w = frame.shape[:2]

    # 좌상단: 성능
    draw_panel(frame, 10, 10, 240, 130)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (22, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Inference: {infer_ms:5.1f} ms", (22, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    status = "PAUSED" if paused else "LIVE"
    status_color = (0, 0, 255) if paused else (0, 255, 0)
    cv2.putText(frame, status, (22, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

    mode_text = "GROUP" if group_mode else "CLASS"
    cv2.putText(frame, f"Mode: {mode_text}", (22, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # 우상단: 그룹별 비율
    if group_ratios:
        panel_w = 250
        items = [("safe", "Safe (sidewalk)"),
                 ("braille", "Braille block"),
                 ("caution", "Caution zone"),
                 ("danger", "Danger (road)"),
                 ("bike", "Bike lane")]
        visible = [(k, n, group_ratios.get(k, 0.0)) for k, n in items
                   if group_ratios.get(k, 0.0) > 0.5]
        panel_h = 44 + 24 * max(len(visible), 1)
        x0 = w - panel_w - 10
        draw_panel(frame, x0, 10, panel_w, panel_h)
        cv2.putText(frame, "Surface Coverage", (x0 + 12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

        for i, (key, name, pct) in enumerate(visible):
            y = 58 + i * 24
            color = GROUP_META[key][1]
            # 색상 도트
            cv2.rectangle(frame, (x0 + 12, y - 10), (x0 + 26, y + 2), color, -1)
            cv2.putText(frame, f"{name:<18s}{pct:5.1f}%",
                        (x0 + 32, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # 하단: 발 앞 영역 + 안내 메시지
    if foot_info:
        draw_foot_zone_box(frame)
        dominant, all_foot = foot_info
        msg, group = generate_guidance(dominant, {})

        # 안내 메시지 박스 (가운데 하단)
        msg_color = GROUP_META.get(group, GROUP_META["neutral"])[1]
        box_h = 60
        box_y = h - box_h - 20
        box_w = min(700, w - 40)
        box_x = (w - box_w) // 2

        # 색상 사이드 바
        cv2.rectangle(frame, (box_x, box_y), (box_x + 8, box_y + box_h),
                      msg_color, -1)
        # 본체
        cv2.rectangle(frame, (box_x + 8, box_y), (box_x + box_w, box_y + box_h),
                      (20, 20, 20), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                      msg_color, 2)

        # ASCII 안전 버전 메시지
        ascii_msg = make_ascii_guidance(dominant)
        cv2.putText(frame, "FOOT ZONE:", (box_x + 22, box_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, ascii_msg, (box_x + 22, box_y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_color, 2, cv2.LINE_AA)

    if help_on:
        help_text = "[Q/ESC]quit [S]save [P]pause [G]group/class [H]help"
        cv2.putText(frame, help_text, (15, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_panel(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)


def draw_foot_zone_box(frame):
    """화면에 '발 앞 영역' 가이드 박스를 표시."""
    h, w = frame.shape[:2]
    y1, y2 = int(h * 0.66), h - 90
    x1, x2 = int(w * 0.25), int(w * 0.75)
    # 점선 효과
    for i in range(x1, x2, 12):
        cv2.line(frame, (i, y1), (min(i + 6, x2), y1), (255, 255, 255), 1)
    for i in range(y1, y2, 12):
        cv2.line(frame, (x1, i), (x1, min(i + 6, y2)), (255, 255, 255), 1)
        cv2.line(frame, (x2, i), (x2, min(i + 6, y2)), (255, 255, 255), 1)


def make_ascii_guidance(dominant):
    """OpenCV에서 깨지지 않는 영어 안내문."""
    if dominant is None:
        return "no detection"
    cls_name, pct = dominant
    group = CLASS_TO_GROUP.get(cls_name, "neutral")
    label = cls_name.replace("_", " ")

    if group == "danger":
        return f"[!] DANGER: {label} ({pct:.0f}%)"
    elif group == "bike":
        return f"[!] BIKE LANE ({pct:.0f}%)"
    elif group == "caution":
        return f"[!] CAUTION: {label} ({pct:.0f}%)"
    elif group == "braille":
        return f"[OK] braille block ({pct:.0f}%)"
    elif group == "safe":
        return f"[OK] sidewalk: {label} ({pct:.0f}%)"
    else:
        return f"{label} ({pct:.0f}%)"


# ============================================
# 7. 메인
# ============================================
def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"❌ 모델 파일을 찾을 수 없습니다: {model_path.resolve()}\n"
            f"   --model <경로> 옵션으로 정확한 경로를 지정하세요."
        )

    print(f"🔄 모델 로드: {model_path}")
    model = YOLO(str(model_path))

    class_names = model.names if hasattr(model, "names") else {}
    n_classes = len(class_names) if class_names else 21
    print(f"✅ 클래스 {n_classes}개 로드됨")

    # 매핑 점검
    unmapped = [name for i, name in class_names.items()
                if name not in CLASS_TO_GROUP]
    if unmapped:
        print(f"⚠️  그룹 미매핑 클래스: {unmapped}")

    class_palette = make_palette(n_classes)

    print(f"📷 웹캠 {args.camera}번 연결 중...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"❌ 웹캠 {args.camera}번을 열 수 없습니다.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps_buffer = deque(maxlen=30)
    last_time = time.time()

    capture_dir = Path("captures")
    capture_dir.mkdir(exist_ok=True)

    paused = False
    help_on = True
    group_mode = True
    last_display = None
    last_stats = None

    # 콘솔 안내문 쿨다운 (너무 자주 출력 방지)
    last_console_group = None
    last_console_time = 0

    print("\n🎬 테스트 시작!")
    print("   [Q/ESC]종료  [S]캡처  [P]일시정지  [G]그룹/클래스  [H]도움말\n")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  프레임 읽기 실패")
                    break
                frame = cv2.flip(frame, 1)

                t0 = time.time()
                results = model.predict(frame, conf=args.conf,
                                        imgsz=args.imgsz, verbose=False)
                infer_ms = (time.time() - t0) * 1000
                result = results[0]

                display, ratios, group_ratios = render_segmentation(
                    frame, result, class_names,
                    group_mode=group_mode, class_palette=class_palette
                )

                foot_info = analyze_foot_zone(result, class_names, frame.shape)

                now = time.time()
                fps_buffer.append(1.0 / max(now - last_time, 1e-6))
                last_time = now
                fps = sum(fps_buffer) / len(fps_buffer)

                last_display = display
                last_stats = (fps, infer_ms, group_ratios, foot_info)

                # 콘솔에 발 앞 상태 변경 시 한국어로 출력
                if foot_info and foot_info[0]:
                    dom_name = foot_info[0][0]
                    group = CLASS_TO_GROUP.get(dom_name, "neutral")
                    if group != last_console_group and (now - last_console_time) > 1.5:
                        kr = KR_LABEL.get(dom_name, dom_name)
                        icon = {"danger": "⚠️", "bike": "🚴", "caution": "⚠️",
                                "braille": "🦯", "safe": "✓", "neutral": "·"}[group]
                        print(f"  {icon} [{group.upper():7s}] {kr}")
                        last_console_group = group
                        last_console_time = now

            if last_display is None:
                time.sleep(0.05)
                continue

            display = last_display.copy()
            if last_stats:
                fps, infer_ms, group_ratios, foot_info = last_stats
                draw_hud(display, fps, infer_ms, group_ratios, foot_info,
                         paused=paused, help_on=help_on, group_mode=group_mode)

            cv2.imshow("VisionGuide - Surface Segmentation Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = capture_dir / f"capture_{ts}.jpg"
                cv2.imwrite(str(out_path), display)
                print(f"💾 저장: {out_path}")
            elif key == ord("p"):
                paused = not paused
                print(f"  {'⏸️  일시정지' if paused else '▶️  재개'}")
            elif key == ord("g"):
                group_mode = not group_mode
                print(f"  🎨 모드 전환: {'그룹 색상' if group_mode else '클래스 색상'}")
            elif key == ord("h"):
                help_on = not help_on

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 테스트 종료")


if __name__ == "__main__":
    main()

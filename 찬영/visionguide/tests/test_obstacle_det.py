"""
VisionGuide - Obstacle Detection 웹캠 실시간 테스트

29개 클래스를 시각장애인 관점의 5개 그룹으로 재분류:
  🚗 VEHICLE     - 차량 (차, 버스, 트럭, 오토바이) → 높은 위험
  🚴 MICRO       - 개인형 이동장치 (자전거, 스쿠터) → 빠른 접근 위험
  🚶 PERSON      - 사람, 동물, 유모차, 휠체어 → 피해야 할 대상
  🚏 FIXED       - 고정 장애물 (볼라드, 가로수, 벤치 등) → 부딪힘 주의
  🚦 SIGNAL      - 신호등 / 교통표지판 → 색상 판별 대상

사용 예:
    python test_obstacle_det.py --model ai_models/yolo_obstacle/weights/best.pt
    python test_obstacle_det.py --model best.pt --camera 1 --conf 0.4

조작:
    q / ESC : 종료
    s       : 캡처 저장 (captures/)
    p       : 일시정지
    f       : 필터 토글 (전체 / 위험만 / 신호등만)
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
# 1. 그룹 정의 + 색상
# ============================================
# (표시명, BGR 색상, 위험도 레벨 0~3)
GROUP_META = {
    "vehicle": ("VEHICLE", ( 60,  60, 240), 3),   # 빨강 - 최고 위험
    "micro":   ("MICRO",   ( 80, 120, 255), 3),   # 주황빨강 - 빠른 접근
    "person":  ("PERSON",  (255, 180,  80), 2),   # 하늘색 - 피해야 함
    "fixed":   ("FIXED",   ( 60, 200, 255), 2),   # 주황 - 부딪힘
    "signal":  ("SIGNAL",  (  0, 255, 255), 1),   # 노랑 - 정보
    "other":   ("OTHER",   (180, 180, 180), 0),   # 회색
}

CLASS_TO_GROUP = {
    # 🚗 차량 (VEHICLE) - 충돌 시 치명적
    "car":        "vehicle",
    "bus":        "vehicle",
    "truck":      "vehicle",
    "motorcycle": "vehicle",

    # 🚴 개인형 이동장치 (MICRO) - 빠른 접근, 조용함
    "bicycle":    "micro",
    "scooter":    "micro",

    # 🚶 사람/동물/기타 이동체 (PERSON)
    "person":          "person",
    "cat":             "person",
    "dog":             "person",
    "stroller":        "person",   # 유모차
    "wheelchair":      "person",   # 휠체어
    "carrier":         "person",   # 리어카/손수레
    "movable_signage": "person",   # 이동식 홍보물/안내판

    # 🚏 고정 장애물 (FIXED) - 부딪힘 주의
    "barricade":       "fixed",   # 바리케이드
    "bench":           "fixed",
    "bollard":         "fixed",   # 볼라드 (차량 진입 방지 기둥)
    "chair":           "fixed",
    "fire_hydrant":    "fixed",   # 소화전
    "kiosk":           "fixed",   # 키오스크
    "parking_meter":   "fixed",   # 주차요금징수기
    "pole":            "fixed",   # 기둥
    "potted_plant":    "fixed",   # 화분
    "power_controller": "fixed",  # 전력제어함
    "stop":            "fixed",   # 버스/택시 정류장
    "table":           "fixed",
    "tree_trunk":      "fixed",   # 가로수 기둥
    "traffic_light_controller": "fixed",   # 신호등 제어기

    # 🚦 신호 (SIGNAL) - 색상 판별 대상
    "traffic_light":  "signal",
    "traffic_sign":   "signal",
}

# 한국어 표시명
KR_LABEL = {
    # 이동체
    "bicycle": "자전거", "bus": "버스", "car": "승용차",
    "carrier": "리어카", "cat": "고양이", "dog": "개",
    "motorcycle": "오토바이", "movable_signage": "이동식 안내판",
    "person": "사람", "scooter": "스쿠터", "stroller": "유모차",
    "truck": "트럭", "wheelchair": "휠체어",
    # 고정체
    "barricade": "바리케이드", "bench": "벤치", "bollard": "볼라드",
    "chair": "의자", "fire_hydrant": "소화전", "kiosk": "키오스크",
    "parking_meter": "주차요금기", "pole": "기둥", "potted_plant": "화분",
    "power_controller": "전력제어함", "stop": "정류장", "table": "탁자",
    "traffic_light": "신호등", "traffic_light_controller": "신호등제어기",
    "traffic_sign": "교통표지판", "tree_trunk": "가로수",
}


# ============================================
# 2. 파싱
# ============================================
def parse_args():
    parser = argparse.ArgumentParser(description="Obstacle Detection 웹캠 테스트")
    parser.add_argument("--model", "-m",
                        default="ai_models/yolo_obstacle/weights/best.pt",
                        help="학습된 YOLO 모델 경로")
    parser.add_argument("--camera", "-c", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--conf", type=float, default=0.35, help="신뢰도 임계값")
    parser.add_argument("--imgsz", type=int, default=640, help="추론 이미지 크기")
    parser.add_argument("--width", type=int, default=1280, help="웹캠 가로")
    parser.add_argument("--height", type=int, default=720, help="웹캠 세로")
    return parser.parse_args()


# ============================================
# 3. 바운딩 박스 그리기
# ============================================
def draw_detections(frame, result, class_names, filter_mode="all"):
    """
    탐지된 객체에 bbox + 라벨을 그린다.
    filter_mode:
      "all"     - 전체 표시
      "danger"  - 위험군(vehicle, micro, person, fixed)만
      "signal"  - 신호등/표지판만
    """
    if result.boxes is None or len(result.boxes) == 0:
        return frame, {}, []

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    group_counts = defaultdict(int)
    h, w = frame.shape[:2]
    frame_area = h * w

    # 위험도 순으로 정렬 (큰 객체 = 가까운 객체부터 그리기)
    items = []
    for box, cls_id, conf in zip(boxes, classes, confs):
        cls_name = class_names.get(cls_id, f"cls{cls_id}")
        group = CLASS_TO_GROUP.get(cls_name, "other")

        # 필터 적용
        if filter_mode == "danger" and group not in ("vehicle", "micro", "person", "fixed"):
            continue
        if filter_mode == "signal" and group != "signal":
            continue

        x1, y1, x2, y2 = box.astype(int)
        area = (x2 - x1) * (y2 - y1)
        area_pct = (area / frame_area) * 100

        items.append({
            "cls_name": cls_name,
            "cls_id": cls_id,
            "group": group,
            "bbox": (x1, y1, x2, y2),
            "conf": float(conf),
            "area_pct": area_pct,
            "center_y": (y1 + y2) // 2,
        })
        group_counts[group] += 1

    # 작은 것 먼저, 큰 것 나중에 (큰 박스가 위에 그려져서 잘 보이게)
    items.sort(key=lambda x: x["area_pct"])

    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        color = GROUP_META[item["group"]][1]
        conf = item["conf"]

        # 박스 두께를 면적 비율에 비례 (가까운 건 굵게)
        thickness = 2 if item["area_pct"] < 5 else 3 if item["area_pct"] < 15 else 4

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 라벨 (이름 + 신뢰도)
        label = f"{item['cls_name']} {conf:.2f}"
        draw_label(frame, label, (x1, y1), color)

    return frame, dict(group_counts), items


def draw_label(img, text, top_left, color):
    """bbox 위에 라벨 박스를 그린다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = top_left
    # 라벨이 화면 위로 삐져나가면 박스 안쪽으로
    y_label = y - 8 if y - th - 10 > 0 else y + th + 10
    x1, y1 = x, y_label - th - 6
    x2, y2 = x + tw + 8, y_label + 2

    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.putText(img, text, (x + 4, y_label - 2),
                font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


# ============================================
# 4. 근접 위험 판단 (시각장애인 안내 로직)
# ============================================
def assess_proximity_risk(items, frame_shape):
    """
    현재 프레임에서 가장 시급한 위험 객체를 판단.
    - 화면 하단(=가까움) + 큰 면적(=가까움) + 높은 위험도 = 최우선
    """
    if not items:
        return None

    h, w = frame_shape[:2]

    def risk_score(item):
        group = item["group"]
        group_weight = {"vehicle": 10, "micro": 9, "person": 6,
                        "fixed": 4, "signal": 2, "other": 1}.get(group, 1)

        # 화면 하단에 있을수록 가까움
        y_weight = (item["center_y"] / h) ** 2

        # 면적이 클수록 가까움
        area_weight = min(item["area_pct"] / 20, 1.0)  # 20% 이상은 최대치

        return group_weight * (y_weight + area_weight)

    ranked = sorted(items, key=risk_score, reverse=True)
    return ranked[0] if ranked else None


def generate_alert(top_item):
    """가장 위험한 객체에 대한 안내 메시지."""
    if top_item is None:
        return "no detection", "other"

    cls_name = top_item["cls_name"]
    group = top_item["group"]
    area = top_item["area_pct"]

    kr = KR_LABEL.get(cls_name, cls_name)
    label = cls_name.replace("_", " ")

    # 면적 기준 거리 추정 (Week 3에 Depth로 교체 예정)
    if area > 15:
        distance = "VERY CLOSE"
    elif area > 5:
        distance = "NEARBY"
    else:
        distance = "AHEAD"

    if group == "vehicle":
        msg = f"[!] {distance}: {label} (vehicle)"
    elif group == "micro":
        msg = f"[!] {distance}: {label} approaching"
    elif group == "person":
        msg = f"[i] {distance}: {label}"
    elif group == "fixed":
        msg = f"[!] {distance}: {label} obstacle"
    elif group == "signal":
        msg = f"[?] {label} detected - check color"
    else:
        msg = f"{label} ({distance})"

    return msg, group


# ============================================
# 5. HUD 패널
# ============================================
def draw_hud(frame, fps, infer_ms, group_counts, top_item, filter_mode,
             paused=False, help_on=True):
    h, w = frame.shape[:2]

    # 좌상단: 성능
    draw_panel(frame, 10, 10, 240, 150)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (22, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Inference: {infer_ms:5.1f} ms", (22, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    status = "PAUSED" if paused else "LIVE"
    status_color = (0, 0, 255) if paused else (0, 255, 0)
    cv2.putText(frame, status, (22, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Filter: {filter_mode.upper()}", (22, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    total = sum(group_counts.values())
    cv2.putText(frame, f"Total: {total} objects", (22, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # 우상단: 그룹별 카운트
    if group_counts:
        panel_w = 220
        items_order = [("vehicle", "Vehicle"),
                       ("micro",   "Micro-mobility"),
                       ("person",  "Person/Pet"),
                       ("fixed",   "Fixed obstacle"),
                       ("signal",  "Signal/Sign")]
        visible = [(k, n, group_counts.get(k, 0)) for k, n in items_order
                   if group_counts.get(k, 0) > 0]

        panel_h = 44 + 24 * max(len(visible), 1)
        x0 = w - panel_w - 10
        draw_panel(frame, x0, 10, panel_w, panel_h)
        cv2.putText(frame, "Detected Objects", (x0 + 12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

        for i, (key, name, count) in enumerate(visible):
            y = 58 + i * 24
            color = GROUP_META[key][1]
            cv2.rectangle(frame, (x0 + 12, y - 10), (x0 + 26, y + 2), color, -1)
            cv2.putText(frame, f"{name:<16s}{count:>2d}",
                        (x0 + 32, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # 하단: 최우선 위험 알림
    if top_item:
        msg, group = generate_alert(top_item)
        msg_color = GROUP_META.get(group, GROUP_META["other"])[1]

        box_h = 60
        box_y = h - box_h - 20
        box_w = min(700, w - 40)
        box_x = (w - box_w) // 2

        cv2.rectangle(frame, (box_x, box_y), (box_x + 8, box_y + box_h),
                      msg_color, -1)
        cv2.rectangle(frame, (box_x + 8, box_y), (box_x + box_w, box_y + box_h),
                      (20, 20, 20), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                      msg_color, 2)

        cv2.putText(frame, "TOP RISK:", (box_x + 22, box_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, msg, (box_x + 22, box_y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_color, 2, cv2.LINE_AA)

    if help_on:
        help_text = "[Q/ESC]quit [S]save [P]pause [F]filter [H]help"
        cv2.putText(frame, help_text, (15, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_panel(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)


# ============================================
# 6. 메인
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
    n_classes = len(class_names) if class_names else 29
    print(f"✅ 클래스 {n_classes}개 로드됨")

    # 매핑 점검
    unmapped = [name for i, name in class_names.items()
                if name not in CLASS_TO_GROUP]
    if unmapped:
        print(f"⚠️  그룹 미매핑 클래스: {unmapped}")

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
    filter_modes = ["all", "danger", "signal"]
    filter_idx = 0
    last_display = None
    last_stats = None

    last_console_group = None
    last_console_time = 0

    print("\n🎬 테스트 시작!")
    print("   [Q/ESC]종료  [S]캡처  [P]일시정지  [F]필터전환  [H]도움말\n")

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

                display, group_counts, items = draw_detections(
                    frame.copy(), result, class_names,
                    filter_mode=filter_modes[filter_idx]
                )

                top_item = assess_proximity_risk(items, frame.shape)

                now = time.time()
                fps_buffer.append(1.0 / max(now - last_time, 1e-6))
                last_time = now
                fps = sum(fps_buffer) / len(fps_buffer)

                last_display = display
                last_stats = (fps, infer_ms, group_counts, top_item)

                # 콘솔에 TOP RISK 변경 시 한글 출력
                if top_item:
                    group = top_item["group"]
                    cls_name = top_item["cls_name"]
                    key = f"{group}:{cls_name}"
                    if key != last_console_group and (now - last_console_time) > 1.5:
                        kr = KR_LABEL.get(cls_name, cls_name)
                        icon = {"vehicle": "🚗", "micro": "🚴",
                                "person": "🚶", "fixed": "🚏",
                                "signal": "🚦", "other": "·"}[group]
                        area_str = f"{top_item['area_pct']:.1f}%"
                        print(f"  {icon} [{group.upper():7s}] {kr:20s} (크기 {area_str})")
                        last_console_group = key
                        last_console_time = now

            if last_display is None:
                time.sleep(0.05)
                continue

            display = last_display.copy()
            if last_stats:
                fps, infer_ms, group_counts, top_item = last_stats
                draw_hud(display, fps, infer_ms, group_counts, top_item,
                         filter_mode=filter_modes[filter_idx],
                         paused=paused, help_on=help_on)

            cv2.imshow("VisionGuide - Obstacle Detection Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = capture_dir / f"obstacle_{ts}.jpg"
                cv2.imwrite(str(out_path), display)
                print(f"💾 저장: {out_path}")
            elif key == ord("p"):
                paused = not paused
                print(f"  {'⏸️  일시정지' if paused else '▶️  재개'}")
            elif key == ord("f"):
                filter_idx = (filter_idx + 1) % len(filter_modes)
                print(f"  🔍 필터: {filter_modes[filter_idx]}")
            elif key == ord("h"):
                help_on = not help_on

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 테스트 종료")


if __name__ == "__main__":
    main()

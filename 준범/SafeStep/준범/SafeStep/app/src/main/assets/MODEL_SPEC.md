# SafeStep 모델 사양 (팀원용)

이 문서는 PyTorch Mobile 디텍터에 모델을 끼워넣기 위한 인터페이스 명세다.
현재 앱은 모델 없이도 정상 실행되며 (NoOpDetector), 아래 두 파일만 이 폴더에 떨어뜨리면 자동으로 PyTorchDetector 가 활성화된다.

## 필요한 파일

| 파일명          | 위치                          | 설명 |
|-----------------|-------------------------------|------|
| `model.ptl`     | `app/src/main/assets/model.ptl` | TorchScript Lite 형식 (`*.ptl`). `_save_for_lite_interpreter()` 로 export. |
| `labels.txt`    | `app/src/main/assets/labels.txt` | 한 줄당 클래스 1개. 인덱스 = 모델 출력 클래스 인덱스. |

> ⚠ `.pt` (full TorchScript) 가 아닌 `.ptl` (lite) 이어야 한다. 모바일에서 추론 실패한다.

## 모델 export 예시 (Python)

```python
import torch
from your_model import build_model

model = build_model().eval()
example = torch.rand(1, 3, 640, 640)
traced = torch.jit.trace(model, example)
traced._save_for_lite_interpreter("model.ptl")

# 라벨 파일
with open("labels.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))   # 80개 클래스라면 80줄
```

## 입력 사양 (앱 측 기본값)

- 입력 크기: **640 × 640**
- 채널 순서: **RGB**
- 정규화: ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- dtype: float32

기본값은 `ObjectDetector.kt` → `PyTorchDetector(inputSize = 640)` 에 있다. 다르면 알려주거나 PR 로 바꿔도 됨.

## 출력 포맷 (확정 필요!)

이 부분이 모델마다 달라서 `parseRawOutput()` 을 채울 때 필요하다. 둘 중 하나로 통일하자:

### 옵션 A: 원시 YOLO 출력
```
shape: [1, N, 5 + num_classes]
각 행: [cx, cy, w, h, obj_conf, cls0_score, cls1_score, ...]
좌표는 입력 픽셀 단위 (0..640)
```
앱에서 NMS / threshold 직접 처리.

### 옵션 B: 후처리 포함 출력 (튜플)
```
(boxes: [K, 4]   xyxy 픽셀,
 scores: [K],
 labels: [K]   int)
```
앱은 그대로 받아서 화면 비율로 정규화만 하면 됨. **이쪽이 더 편함.**

## 클래스 라벨 가이드

시각장애인용 안내가 자연스럽게 나오도록, 가능하면 다음 클래스를 포함:

- `person`, `car`, `truck`, `bus`, `bicycle`, `motorcycle`
- `chair`, `pole`, `tree`, `stairs`, `door`, `wall`

`MainActivity.kt` 의 `labelToKorean()` 에서 한국어 매핑한다. 새 클래스 추가하면 거기에도 한 줄 추가하면 됨.

## 체크리스트 (모델 인계 시)

- [ ] `model.ptl` 파일 (Lite Interpreter 형식)
- [ ] `labels.txt` 파일
- [ ] 입력 크기, 정규화값이 위 기본값과 같은지 확인
- [ ] 출력 포맷이 옵션 A 인지 B 인지 명시
- [ ] mAP / 추론 속도 (FPS) 대략값

# 트러블슈팅

## 카메라가 열리지 않음

증상:

```text
Camera index 0 could not be opened.
```

해결:

1. USB 연결을 다시 확인합니다.
2. macOS 카메라 권한을 확인합니다.
3. camera index를 확인합니다.

```bash
python scripts/inspect_camera.py --max-index 6
```

macOS에서 문제가 계속되면 backend를 명시합니다.

```bash
python scripts/direct_red_cube_tracker.py --camera 0 --backend avfoundation
```

## Innomaker 1080p 카메라가 1920x1080으로 안 잡힘

일부 UVC 카메라는 요청한 해상도를 그대로 쓰지 않을 수 있습니다. `inspect_camera.py` 출력의 `actual=...` 값을 확인합니다.

수업에서는 1280x720도 충분합니다.

```bash
python scripts/direct_red_cube_tracker.py --camera 0 --width 1280 --height 720
```

## 빨간 큐브가 OpenCV 방식에서 잘 안 잡힘

원인:

- 조명이 너무 어둡거나 강함
- 큐브가 반사되어 흰색처럼 보임
- 배경에 빨간색이 많음
- HSV threshold가 현재 조명과 맞지 않음

해결:

1. `t`를 눌러 HSV tuning 창을 켭니다.
2. `S low`, `V low` 값을 낮추거나 높여봅니다.
3. 빨간색 hue 범위를 조정합니다.
4. 큐브와 배경의 색상 대비를 높입니다.

## End-to-End 모델이 항상 같은 action만 예측함

가장 흔한 원인은 label 불균형입니다.

예:

```text
left: 5
right: 8
up: 4
down: 6
close: 80
```

이 경우 모델은 대부분 `close`를 찍는 것이 loss를 줄이기 쉽습니다.

해결:

- label별 sample 수를 비슷하게 맞춥니다.
- 최소 label별 20장, 가능하면 40장 이상 수집합니다.
- cursor 위치와 큐브 위치를 다양하게 바꿉니다.

## 학습 정확도는 높은데 실시간 추론이 틀림

가능한 원인:

- training data와 실시간 화면의 조명이 다름
- 큐브 위치가 수집 범위 밖임
- cursor state를 다양하게 수집하지 않음
- validation set이 training set과 너무 비슷함

해결:

- 실패한 조건의 데이터를 추가로 수집합니다.
- 다시 학습합니다.
- 조명과 카메라 위치를 고정합니다.

## `torch` 설치가 느리거나 실패함

네트워크 상태에 따라 PyTorch 설치가 오래 걸릴 수 있습니다.

대안:

1. 수업 전에 미리 설치합니다.
2. 설치가 실패하면 OpenCV 직접 방식만 먼저 진행합니다.
3. End-to-End 부분은 강사가 미리 학습한 checkpoint를 준비합니다.

## OpenCV 창에서 키 입력이 안 먹음

OpenCV 창이 활성화되어 있어야 합니다. 마우스로 영상 창을 한 번 클릭한 뒤 키를 누릅니다.

## 데이터 저장 위치를 지우고 새로 시작하고 싶음

기존 dataset을 보존하려면 새 이름을 쓰는 것이 안전합니다.

```bash
python scripts/collect_e2e_dataset.py --dataset data/e2e_red_cube_round2
python scripts/train_e2e_policy.py --dataset data/e2e_red_cube_round2 --model-out models/e2e_red_cube_policy_round2.pt
```


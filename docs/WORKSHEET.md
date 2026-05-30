# 학생용 실습지

## 오늘의 질문

```text
로봇은 빨간 큐브가 어디 있는지 어떻게 알 수 있을까?
```

## 실습 1: OpenCV 직접 방식

실행:

```bash
python scripts/direct_red_cube_tracker.py --camera 0 --width 1920 --height 1080 --fps 30 --show-mask
```

관찰한 값:

```text
center_x =
center_y =
area =
```

### 질문

1. 큐브를 왼쪽으로 옮기면 `center_x`는 어떻게 변했나요?
2. 큐브를 카메라에 가까이 가져가면 `area`는 어떻게 변했나요?
3. 손으로 그림자를 만들었을 때 mask는 안정적이었나요?
4. 빨간색 다른 물체를 넣으면 tracker는 무엇을 큐브로 보나요?

## 실습 2: End-to-End Dataset 수집

실행:

```bash
python scripts/collect_e2e_dataset.py --camera 0 --dataset data/e2e_red_cube --width 1280 --height 720
```

키:

| 키 | 의미 |
|---|---|
| `a` | left label 저장 |
| `d` | right label 저장 |
| `w` | up label 저장 |
| `s` | down label 저장 |
| `space` | close label 저장 |
| `j/l/i/k` | 가상 cursor 이동 |
| `c` | cursor 중앙 이동 |

내가 모은 sample 수:

```text
left:
right:
up:
down:
close:
```

### 질문

1. 왜 label별 sample 수를 비슷하게 모아야 할까요?
2. 큐브 위치를 항상 같은 곳에 두면 모델은 무엇을 잘못 배울 수 있을까요?
3. 이 실습에서 `cursor_x`, `cursor_y`는 실제 로봇의 어떤 정보와 비슷한가요?

## 실습 3: 모델 학습

실행:

```bash
python scripts/train_e2e_policy.py --dataset data/e2e_red_cube --epochs 12 --model-out models/e2e_red_cube_policy.pt
```

기록:

```text
best validation accuracy =
가장 부족한 label =
가장 많이 모은 label =
```

### 질문

1. training accuracy와 validation accuracy가 많이 다르면 어떤 의미일까요?
2. validation accuracy가 높아도 실제 camera 화면에서 틀릴 수 있는 이유는 무엇일까요?

## 실습 4: 실시간 추론

실행:

```bash
python scripts/run_e2e_policy.py --camera 0 --checkpoint models/e2e_red_cube_policy.pt --width 1280 --height 720
```

관찰:

```text
큐브가 왼쪽에 있을 때 prediction =
큐브가 오른쪽에 있을 때 prediction =
cursor가 큐브 위에 있을 때 prediction =
```

### 실패 사례 실험

아래 조건을 하나씩 바꿔보세요.

| 조건 | 결과 |
|---|---|
| 조명 어둡게 하기 | |
| 빨간 물체 추가하기 | |
| 큐브를 화면 구석에 놓기 | |
| 배경을 복잡하게 바꾸기 | |

## 최종 정리

OpenCV 직접 방식:

```text
image -> 사람이 만든 빨간색 검출 rule -> cube center
```

End-to-End 모방학습:

```text
image + state -> neural network -> action
```

오늘 배운 것을 SO-ARM101에 연결하면:

```text
camera image + robot state -> motor action
```


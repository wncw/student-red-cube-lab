# Red Cube Camera Lab

카메라만 사용해서 두 가지 방식을 비교하는 실습입니다.

1. **OpenCV 직접 방식**
   - 빨간색을 HSV threshold로 찾습니다.
   - 빨간 큐브의 bounding square와 중심 좌표를 실시간으로 표시합니다.

2. **End-to-End 미니 모방학습 방식**
   - 빨간 큐브 좌표를 코드로 직접 계산하지 않습니다.
   - 카메라 이미지와 가상 gripper 위치를 입력으로 받아, 사람이 입력한 action label을 학습합니다.

## 준비물

- 카메라
- 빨간색 정육면체 큐브
- Python 3.9 이상 권장
- macOS, Windows, Linux 중 하나

## 코드 내려받기

### Windows에서 Git 설치

PowerShell을 열고 아래 명령어를 실행합니다.

```powershell
winget install --id Git.Git -e --source winget
```

설치가 끝나면 PowerShell을 완전히 닫고 다시 엽니다. 그다음 Git이 설치되었는지 확인합니다.

```powershell
git --version
```

`git version ...`처럼 버전이 나오면 정상입니다.

설치했는데도 `git` 명령어가 인식되지 않으면 PowerShell을 완전히 닫고 다시 열어보세요. 그래도 안 되면 [Git 설치 후 명령어가 인식되지 않음](docs/TROUBLESHOOTING.md#git-설치-후-명령어가-인식되지-않음)을 확인합니다.

### Git으로 내려받기

Windows PowerShell, macOS Terminal, Linux Terminal에서 아래 명령어를 실행합니다.

```bash
git clone https://github.com/wncw/student-red-cube-lab.git
cd student-red-cube-lab
```

### Git 설치가 안 될 때

수업 중 Git 설치가 오래 걸리면 ZIP으로 받아도 됩니다.

1. https://github.com/wncw/student-red-cube-lab 접속
2. 초록색 `Code` 버튼 클릭
3. `Download ZIP` 클릭
4. 압축 해제
5. 압축을 푼 폴더에서 Terminal 또는 PowerShell 실행

## Python 환경 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell에서는 아래 명령어를 사용합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`Activate.ps1` 실행 중 `PSSecurityException` 보안 오류가 나오면 아래 명령어를 먼저 실행한 뒤 다시 activate 합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

그래도 막히면 activate 없이 `.venv`의 Python을 직접 실행할 수 있습니다. 자세한 방법은 [PowerShell에서 Activate.ps1 보안 오류가 발생함](docs/TROUBLESHOOTING.md#powershell에서-activateps1-보안-오류가-발생함)을 확인합니다.

## 1. 카메라 번호 확인

```bash
python scripts/inspect_camera.py --max-index 6
```

정상적으로 보이는 camera index를 기억합니다. 아래 예시는 `--camera 0`을 사용하지만, 본인 컴퓨터에서 다른 번호라면 바꿔야 합니다.

## 2. OpenCV 직접 방식 실행

```bash
python scripts/direct_red_cube_tracker.py --camera 0 --width 1280 --height 720 --fps 30 --show-mask
```

키:

| 키 | 동작 |
|---|---|
| `q` 또는 `ESC` | 종료 |
| `s` | screenshot 저장 |
| `t` | HSV tuning 창 켜기/끄기 |
| `m` | mask 화면 켜기/끄기 |
| `g` | 가상 gripper 이동 화살표 켜기/끄기 |

관찰할 것:

- 큐브를 움직이면 중심 좌표가 어떻게 바뀌는가
- 큐브가 화면 중앙보다 왼쪽/오른쪽/위/아래에 있을 때 action 화살표가 어떻게 바뀌는가
- 조명이 바뀌면 mask가 어떻게 바뀌는가
- 빨간색 다른 물체를 넣으면 무엇을 큐브로 착각하는가

화면 중앙의 십자는 가상의 gripper 목표점입니다. 빨간 큐브가 중앙보다 왼쪽에 있으면 `MOVE LEFT`, 오른쪽에 있으면 `MOVE RIGHT`처럼 표시됩니다. 중앙 허용 범위 안에 들어오면 `CENTERED - CLOSE`가 표시됩니다.

## 3. End-to-End Dataset 수집

```bash
python scripts/collect_e2e_dataset.py --camera 0 --dataset data/e2e_red_cube --width 1280 --height 720
```

키:

| 키 | 의미 |
|---|---|
| `a` | `left` label 저장 |
| `d` | `right` label 저장 |
| `w` | `up` label 저장 |
| `s` | `down` label 저장 |
| `space` | `close` label 저장 |
| `j` / `l` | 가상 cursor 왼쪽/오른쪽 이동 |
| `i` / `k` | 가상 cursor 위/아래 이동 |
| `c` | cursor 중앙으로 이동 |
| `q` 또는 `ESC` | 종료 |

권장 수집량:

```text
left   40장 이상
right  40장 이상
up     40장 이상
down   40장 이상
close  40장 이상
```

시간이 부족하면 label별 20장 정도로도 실습은 가능합니다.

## 4. 모델 학습

```bash
python scripts/train_e2e_policy.py --dataset data/e2e_red_cube --epochs 20 --model-out models/e2e_red_cube_policy.pt
```

학습이 끝나면 `models/e2e_red_cube_policy.pt` 파일이 생성됩니다.

## 5. 실시간 추론

```bash
python scripts/run_e2e_policy.py --camera 0 --checkpoint models/e2e_red_cube_policy.pt --width 1280 --height 720
```

키:

| 키 | 동작 |
|---|---|
| `j` / `l` | 가상 cursor 왼쪽/오른쪽 이동 |
| `i` / `k` | 가상 cursor 위/아래 이동 |
| `c` | cursor 중앙으로 이동 |
| `p` | 예측 action으로 cursor 자동 이동 켜기/끄기 |
| `q` 또는 `ESC` | 종료 |

## 핵심 비교

OpenCV 직접 방식:

```text
image -> HSV red mask -> contour -> cube center
```

End-to-End 방식:

```text
image + virtual gripper state -> neural network -> action
```

End-to-End 방식에서는 `cube_x`, `cube_y`를 직접 계산하지 않습니다. 모델이 이미지와 사람이 입력한 action label 사이의 관계를 학습합니다.

## 실습지

실습 중 기록할 내용은 [docs/WORKSHEET.md](docs/WORKSHEET.md)를 사용하세요.

문제가 생기면 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)를 먼저 확인하세요.

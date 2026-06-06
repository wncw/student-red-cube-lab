# 트러블슈팅

## Git 설치 후 명령어가 인식되지 않음

증상:

```text
git : The term 'git' is not recognized...
```

또는:

```text
'git'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램...
```

대부분 Git은 설치됐지만 Windows `PATH`에 Git 실행 파일 위치가 등록되지 않은 상태입니다.

### 1. Git 설치 여부 확인

PowerShell에서 실행합니다.

```powershell
winget list --id Git.Git
```

`Git.Git` 항목이 나오면 설치되어 있는 상태입니다.

### 2. Git 실행 파일 위치 확인

```powershell
Test-Path "C:\Program Files\Git\cmd\git.exe"
```

`True`가 나오면 Git 실행 파일은 있습니다.

직접 실행해서 버전을 확인할 수 있습니다.

```powershell
& "C:\Program Files\Git\cmd\git.exe" --version
```

버전이 나오면 설치는 정상이고 `PATH`만 문제입니다.

### 3. PATH에 Git 추가

PowerShell에서 아래 명령어를 실행합니다.

```powershell
$gitPath = "C:\Program Files\Git\cmd"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath -notlike "*$gitPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$gitPath", "User")
}
```

그다음 PowerShell을 완전히 닫고 새로 연 뒤 확인합니다.

```powershell
git --version
```

### 4. 그래도 안 되면 ZIP으로 진행

수업 중이면 Git 문제로 시간을 오래 쓰지 말고 ZIP 다운로드로 진행해도 됩니다.

1. https://github.com/wncw/student-red-cube-lab 접속
2. 초록색 `Code` 버튼 클릭
3. `Download ZIP` 클릭
4. 압축 해제
5. 압축을 푼 폴더에서 PowerShell 실행

## PowerShell에서 Activate.ps1 보안 오류가 발생함

증상:

```text
PSSecurityException
```

또는:

```text
running scripts is disabled on this system
```

Windows PowerShell의 실행 정책 때문에 가상환경 활성화 스크립트가 막힌 상태입니다. Python 가상환경이 잘못 만들어진 것이 아닙니다.

### 방법 1. 현재 PowerShell 창에서만 임시 허용

PowerShell에서 아래 명령어를 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

그다음 다시 가상환경을 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

이 설정은 현재 PowerShell 창에서만 적용됩니다. 창을 닫으면 원래 상태로 돌아갑니다.

### 방법 2. activate 없이 바로 실행

실행 정책을 바꾸고 싶지 않으면 가상환경 안의 Python을 직접 사용해도 됩니다.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\inspect_camera.py --max-index 6
```

이후 다른 script도 같은 방식으로 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts\direct_red_cube_tracker.py --camera 0 --width 1280 --height 720 --show-mask
```

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

## 카메라 해상도가 1280x720으로 안 잡힘

일부 UVC 카메라는 요청한 해상도를 그대로 쓰지 않을 수 있습니다. `inspect_camera.py` 출력의 `actual=...` 값을 확인합니다.

수업 기본값은 1280x720입니다. 실제 카메라가 다른 해상도로 잡혀도 화면이 정상적으로 나오면 그대로 진행할 수 있습니다.

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

## Dataset을 새로 만들었는데 label count가 누적됨

원인:

- `images/` 폴더만 지우고 `labels.csv`가 남아 있음
- 새 이미지 파일명이 `sample_000000.jpg`부터 다시 만들어지면서 예전 label row가 새 이미지에 연결됨
- 같은 image path가 `labels.csv`에 여러 번 들어감

해결:

최신 코드에서는 `collect_e2e_dataset.py`가 기본적으로 전체 dataset 폴더를 초기화하고 새로 수집합니다.

```bash
python scripts/collect_e2e_dataset.py --camera 0 --dataset data/e2e_red_cube --width 1280 --height 720
```

기존 dataset에 이어서 추가 수집할 때만 `--append`를 붙입니다.

```bash
python scripts/collect_e2e_dataset.py --camera 0 --dataset data/e2e_red_cube --width 1280 --height 720 --append
```

이미 꼬인 dataset은 학습에 쓰지 말고 기본 명령으로 새로 수집하는 것이 안전합니다.

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

## Windows에서 torch WinError 1114가 발생함

증상:

```text
OSError: [WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다
```

경로에 `torch`가 보이면 PyTorch DLL 로딩 문제일 가능성이 큽니다. 이 실습은 GPU가 필요 없으므로 CPU 전용 PyTorch로 재설치하는 것이 가장 안정적입니다.

PowerShell에서 프로젝트 폴더 기준으로 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip cache purge
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

설치 확인:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('torch ok')"
```

그다음 다시 학습합니다.

```powershell
.\.venv\Scripts\python.exe scripts\train_e2e_policy.py --dataset data\e2e_red_cube --epochs 20 --model-out models\e2e_red_cube_policy.pt
```

그래도 같은 오류가 나면 Microsoft Visual C++ Redistributable x64를 설치한 뒤 PowerShell을 새로 열어 다시 확인합니다.

https://aka.ms/vs/17/release/vc_redist.x64.exe

Python 버전도 확인합니다.

```powershell
.\.venv\Scripts\python.exe --version
```

권장 버전은 Python 3.9~3.11입니다.

## OpenCV 창에서 키 입력이 안 먹음

OpenCV 창이 활성화되어 있어야 합니다. 마우스로 영상 창을 한 번 클릭한 뒤 키를 누릅니다.

## 데이터 저장 위치를 지우고 새로 시작하고 싶음

기존 dataset을 보존하려면 새 이름을 쓰는 것이 안전합니다.

```bash
python scripts/collect_e2e_dataset.py --dataset data/e2e_red_cube_round2
python scripts/train_e2e_policy.py --dataset data/e2e_red_cube_round2 --model-out models/e2e_red_cube_policy_round2.pt
```

# Mini Quoridor — Pygame + 직접 학습하는 PyTorch AI

5×5 쿼리도. **외부 AI API는 사용하지 않으며**, PyTorch를 이용해 신경망 구조를 직접 정의합니다. 처음에는 랜덤/규칙 기반 상대와 플레이할 수 있고, 학습 후 신경망 상대도 선택할 수 있습니다.

## 0. 필요한 것

- Python **3.10~3.12 권장** (PyTorch 설치 호환성 면에서 편리함)
- `pygame`: 게임 화면
- `torch`: 신경망/학습 (게임만 실행할 경우 필수 아님)
- 표준 라이브러리: `random`, `collections`, `argparse`, `os` 등 (별도 설치 불필요)

## 1. 설치

프로젝트 폴더에서 터미널을 열고:

**Windows PowerShell**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
`py -3.12`가 없으면 설치된 Python에 맞게 `py -m venv .venv`를 사용해도 됩니다.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch 설치가 환경 때문에 실패하면 [PyTorch 공식 설치 페이지](https://pytorch.org/get-started/locally/)에서 본인 OS와 CPU/GPU에 맞는 명령을 확인하세요. **일단 게임만 실행하려면** `python -m pip install pygame`만 해도 됩니다.

## 2. 게임 실행

```bash
python main.py
```

- **이동**: 초록 점 표시된 타일 클릭
- **M**: 말 이동 / 벽 놓기 모드 전환
- **H / V**: 가로 / 세로 벽 모드
- **벽 놓기**: 모드 변경 후 4칸이 모이는 교차점 부근 클릭
- **R**: 재시작
- **1**: 랜덤 AI (초보용)
- **2**: 최단 경로 기반 AI (기본값)
- **3**: 학습한 신경망 AI (`models/mini_quoridor.pth`가 있어야 사용 가능)

## 3. 신경망 학습

```bash
python train.py --episodes 3000
```

학습된 모델은 `models/mini_quoridor.pth`에 저장됩니다. 이후 `python main.py`에서 **3**을 누르면 신경망 AI와 대결할 수 있습니다.

이 프로젝트의 REINFORCE 학습 코드는 **교육용 실험 베이스라인**입니다. 무작위/규칙 기반 상대보다 반드시 강해지지는 않습니다. 학습 중에는 출력되는 P0/P1 승수만으로 모델 실력이 상승했는지 판단하지 말고 별도 평가를 실시하세요.

```bash
python evaluate.py --games 100 --opponent random
python evaluate.py --games 100 --opponent heuristic
```

## 4. 프로젝트 파일

| 파일 | 역할 |
|---|---|
| `game.py` | 규칙, 경로 탐색, 점프, 벽 설치, 승리 판정, 행동 57개 |
| `agents.py` | 랜덤 AI, 최단 경로 휴리스틱 AI |
| `main.py` | Pygame UI, 키보드/마우스 조작 |
| `neural_net.py` | 직접 정의한 MLP 신경망, 상태 인코딩, 불법 행동 마스킹 |
| `train.py` | REINFORCE 기반 Self-Play 학습 루프 |
| `evaluate.py` | 랜덤/휴리스틱 AI와 승률 비교 |
| `tests.py` | 게임 규칙의 기본 테스트 |
| `requirements.txt` | pip 의존성 |

### 신경망 상세

- 입력: 103차원 벡터 (두 말의 위치, 가로/세로 벽, 남은 벽 개수, 현재 턴)
- 은닉층: 128 → 128, ReLU
- 출력: **57개 행동** — 이동 목적지 25칸 + 가로 벽 16곳 + 세로 벽 16곳
- 불가능한 이동/벽 설치는 마스킹하여 선택할 수 없도록 처리
- 학습: 정책 기울기(REINFORCE) + 승패 보상 + 약한 경로 진척 보상

**주의**: 일반 쿼리도의 규칙을 기본적으로 구현했지만 5×5 축소판은 공식 대회 규격이 아닙니다. 학습 루프에는 무승부 방지를 위한 150턴 제한이 있습니다. 게임 화면에서는 실시간 판별 무승부 제한이 없으므로 이동을 반복하면 계속 진행될 수 있습니다.

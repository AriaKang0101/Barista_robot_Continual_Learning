# Barista Robot Continual Learning

고정된 2-DoF 바리스타 로봇에서 **Naive Sequential PPO**와 **PPO-EWC**의 망각을 비교하는 연구 코드입니다. 한 정책을 `Task A → B → C` 순서로 학습하고, 각 단계가 끝날 때 지금까지 배운 모든 Task를 다시 평가합니다.

## 연구 질문

> 같은 도달 작업을 순차 학습할 때, EWC가 일반 PPO의 catastrophic forgetting을 줄이는가?

비교군 `sequential`은 CL 전용 망각 방지 기법이 없는 PPO입니다. 다만 망각을 측정하려면 하나의 모델을 A→B→C로 계속 학습해야 하므로, 독립 PPO 모델 3개가 아니라 **naive sequential fine-tuning**을 비교군으로 사용합니다. 실험군 `ewc`는 동일 PPO에 policy-Fisher EWC만 추가합니다.

## 원본 저장소와 v4 재설계

- 참고 저장소: [ymk2001/Barista_robot_Reinforcement_Learning](https://github.com/ymk2001/Barista_robot_Reinforcement_Learning)
- 고정 commit: `232dd870ff50f988b1168d7630998438964f5423`
- 장면: `safety_rl_2dof.ttt`
- 장면 Git blob: `925ea5fcc4c2ecb8cb93c1d8cc11e68c434c7b53`

원본 장면의 `/FirstTarget`, `/EndTarget` 위치쌍은 0.15 m AND 성공 조건으로 안전하게 도달 가능한 자세를 찾지 못했습니다. 따라서 v4는 그 Target을 Task A로 강제하지 않습니다. 대신 원본 장면·로봇·장애물·속도 행동·거리진전 보상을 유지하면서, 실제 기구학과 충돌검사를 통과한 도달 가능 목표 A/B/C를 생성합니다. 원본 저장소나 원본 `.ttt`를 수정·저장하지 않습니다.

이 연구는 원본 PPO 결과의 정확한 재현이 아니라, 원본 바리스타 작업공간을 기반으로 만든 **재현 가능한 CL benchmark**입니다.

## 프로토콜 v4

| 항목 | 설정 |
|---|---|
| Task | 안전 연결영역 내부의 서로 다른 joint2/EE 목표 위치쌍 A/B/C |
| 목표 생성 | 실제 안전 관절 자세의 forward kinematics 결과 |
| 기본 관측 | 정규화 관절각 2 + 관절속도 2 + 목표 관절각 2 |
| 행동 | joint1/joint2 목표속도, 각 ±0.8 rad/s |
| 보상 | 두 Cartesian 거리의 감소량 합 ×10, 성공 +100, 충돌 −20 |
| 성공 | joint2와 EE가 각각 목표에서 0.15 m 미만, 충돌 없음 |
| 종료 | 성공 또는 충돌, 500 step에서 timeout |
| 학습 시작 | 목표 근처 10%에서 시작하여 예산 80% 시점까지 전체 안전공간으로 확장 |
| 평가 시작 | Task별 near/medium/far 각 10개, 총 30개 고정 자세 |
| 비교 | 동일 Task·seed·학습량의 sequential PPO vs PPO-EWC |

기본 관측을 저차원 state로 정한 이유는 첫 CL 실험에서 영상 인식 실패와 망각을 섞지 않기 위해서입니다. 관절 encoder와 목표 관절값은 실제 로봇에서도 얻을 수 있습니다. 영상까지 사용하는 `image_state_goal` 모드는 후속 ablation으로 남겨 두었습니다.

## 실패 요인 방지 설계

`prepare`는 다음 조건을 모두 만족해야 task 파일을 작성합니다.

1. 원본 joint 범위에서 17×17 grid를 검사합니다.
2. 링크–벽/머신 충돌, 비인접 자기충돌, 바닥 관통을 제거합니다.
3. 장애물과 1 cm 이상 떨어진 자세만 학습 시작영역에 포함합니다.
4. 인접 grid 사이도 최대 2° 간격으로 검사해 하나의 연결 graph를 만듭니다.
5. A/B/C는 graph 내부이며 이웃 연결도가 높고, 장애물에서 3 cm 이상 떨어진 자세로 선택합니다.
6. 세 목표의 성공영역이 서로 겹치지 않게 선택합니다.
7. 평가 시작점은 각 목표까지 graph 거리로 near/medium/far로 나눕니다.
8. 평가 시작점에서 목표까지 실제 물리 step 제어기가 성공한 경우에만 저장합니다.
9. 두 방법은 동일한 task JSON과 고정 평가 자세를 사용합니다.

검증 제어기는 도달 가능성 검사에만 사용하며 PPO의 행동 라벨이나 demonstration으로 사용하지 않습니다.

## 설치

권장 환경은 Python 3.10, CoppeliaSim 4.9.0 Edu입니다.

```bat
git clone https://github.com/AriaKang0101/Barista_robot_Continual_Learning.git
cd Barista_robot_Continual_Learning
conda create -n barista-cl python=3.10 -y
conda activate barista-cl
```

환경에 맞는 PyTorch를 먼저 설치한 후 나머지 패키지를 설치합니다.

```bat
python -m pip install -r requirements-dev.txt
python -m barista_cl doctor
python -m pytest -q
```

## 1단계: 원본 장면 준비

```bat
python -m barista_cl fetch-scene
```

`scenes/safety_rl_2dof.ttt`의 hash가 원본과 일치해야 합니다. CoppeliaSim GUI를 실행하고 ZeroMQ Remote API 포트 `23000`을 사용하세요. 명령 실행 중 GUI에서 장면이나 로봇을 조작하지 마세요.

## 2단계: v4 Task 생성

```bat
python -m barista_cl prepare --output artifacts/tasks_v4.json
```

기본적으로 Task마다 평가 자세 30개를 생성하므로 시간이 걸립니다. 성공하면 다음 메시지가 출력됩니다.

```text
Prepared protocol v4: 3 reachable goals and 30 starts per task
```

기존 `tasks_v2.json`, `tasks_v3.json` 및 과거 run은 삭제할 필요가 없습니다. v4 실험에서는 `tasks_v4.json`만 사용합니다.

## 3단계: 도달 경로 확인

```bat
python -m barista_cl preview --task A
python -m barista_cl preview --task B
python -m barista_cl preview --task C
```

각 명령이 `(True, ..., 'goal')`로 끝나는지 확인합니다. 이것은 PPO 성공 결과가 아니라 물리적 도달 가능성 검사입니다.

## 4단계: 실행 연결 smoke test

```bat
python -m barista_cl train --method sequential --only-task A --config configs/smoke_v4.json --output runs/v4_smoke_A
```

1,024 step은 연결과 파일 출력을 확인하는 용도입니다. 성공률 0이어도 학습 실패로 판단하지 않습니다.

## 5단계: Task별 learnability pilot

먼저 A를 전체 예산으로 학습합니다.

```bat
python -m barista_cl train --method sequential --only-task A --output runs/v4_pilot_A
```

결과를 확인합니다.

```bat
type runs\v4_pilot_A\metrics.json
```

`final_success_by_band`에서 near뿐 아니라 medium/far도 학습되는지 확인합니다. A가 학습되지 않으면 CL 비교로 넘어가지 않습니다. A가 성공하면 B/C도 각각 새 모델로 확인합니다.

```bat
python -m barista_cl train --method sequential --only-task B --output runs/v4_pilot_B
python -m barista_cl train --method sequential --only-task C --output runs/v4_pilot_C
```

## 6단계: Sequential PPO vs PPO-EWC

한 seed의 첫 비교:

```bat
python -m barista_cl train --method sequential --seed 0 --output runs/v4_sequential_seed0
python -m barista_cl train --method ewc --seed 0 --output runs/v4_ewc_seed0
python -m barista_cl compare runs/v4_sequential_seed0 runs/v4_ewc_seed0 --output artifacts/v4_comparison_seed0
```

최종 연구 결과는 최소 3개 seed를 사용합니다.

```bat
python scripts/run_comparison.py --seeds 0 1 2 --output runs/v4_comparison
```

하나의 CoppeliaSim 인스턴스에서는 반드시 직렬 실행합니다.

## 학습 단계와 평가 행렬

| 완료 단계 | 이번에 학습한 Task | 평가 Task |
|---|---|---|
| 1 | A | A |
| 2 | B | A, B |
| 3 | C | A, B, C |

Task 전환 시 정책과 optimizer를 유지합니다. `sequential`은 그대로 fine-tuning하고, `ewc`는 과거 Task의 policy Fisher와 parameter anchor를 누적하여 PPO gradient에 EWC gradient를 추가합니다. EWC 계산을 위한 추가 환경 rollout은 사용하지 않습니다.

## 기본 학습 설정

| 설정 | 기본값 |
|---|---:|
| timesteps_per_task | 100352 |
| n_steps / batch_size / n_epochs | 512 / 128 / 10 |
| learning rate | Task마다 0.0003 → 0.00005 |
| gamma / gae_lambda | 0.99 / 0.95 |
| ent_coef | 0.01 |
| EWC lambda | 100 |
| Fisher samples | 128 |
| curriculum | 10% 시작, 예산 80%에서 전체공간 |
| 관측 | state_goal |

`ewc_lambda=100`은 시작값이지 최적값이 아닙니다. 최종 평가 seed 0/1/2를 보면서 lambda를 선택하면 안 됩니다. 별도 pilot seed에서 10/100/1000을 비교한 뒤 값을 고정해야 합니다.

## 결과 파일

| 파일 | 내용 |
|---|---|
| `tasks.json`, `config.json` | 실제 사용한 목표·평가 자세·설정 |
| `manifest.json` | seed, 패키지, 장치, 완료 상태, 실제 step |
| `training_episodes.csv` | 학습 episode 성공·충돌·길이 |
| `evaluation_episodes.csv` | Task, near/medium/far, 성공·충돌·timeout·오차 |
| `success_matrix.csv` | 각 학습 단계 후 Task별 성공률 |
| `metrics.json` | 최종 성공률, forgetting, BWT, 거리 band별 성능 |
| `stage_*.zip` | 단계별 정책·optimizer·EWC 상태 |
| `tensorboard/` | PPO와 EWC 로그 |

주요 CL 지표:

- Final mean success: C까지 학습한 뒤 A/B/C 평균 성공률
- Forgetting: 각 과거 Task의 최고 성공률에서 최종 성공률이 감소한 정도
- BWT: 과거 Task를 처음 학습한 직후와 최종 성능의 차이

성공률만 보지 말고 collision과 timeout도 함께 해석해야 합니다.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `barista_cl/simulator.py` | 원본 장면 hash, ZeroMQ, 관절·충돌·물리 step |
| `barista_cl/prepare.py` | 안전 graph, A/B/C, curriculum, 평가 자세 생성 |
| `barista_cl/env.py` | 공통 PPO 환경과 state/image 관측 |
| `barista_cl/ewc.py` | policy-Fisher EWC |
| `barista_cl/experiment.py` | 순차 학습, 평가, checkpoint, 지표 |
| `barista_cl/report.py` | paired seed 비교와 retention graph |
| `barista_cl/core.py` | protocol 검증과 CL 지표 |

## 해석상 주의

- A/B/C는 픽업·추출·서빙 같은 서로 다른 의미 행동이 아니라 서로 다른 목표 위치쌍입니다.
- 목표 관절값이 입력되는 Task-IL/goal-conditioned 설정입니다. 목표 간 positive transfer가 크면 EWC의 이득이 작거나 없을 수도 있으며, 그것도 유효한 결과입니다.
- EWC가 항상 좋아야 하는 것은 아닙니다. 현재 Task 학습을 과도하게 방해하면 최종 평균 성능이 낮아질 수 있습니다.
- state 기반 비교가 안정된 뒤에만 `image_state_goal`을 perception ablation으로 추가하는 것을 권합니다.

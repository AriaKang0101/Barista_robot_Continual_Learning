# Barista Robot Continual Learning

고정된 바리스타 로봇 작업공간에서 **Sequential PPO vs PPO-EWC**를 비교합니다.
한 정책을 `A → B → C`로 순차 학습하고 각 단계 후 이전 작업을 다시 평가합니다.

**현재 상태:** Windows CoppeliaSim 4.9.0에서 v2/v2.1 장면 연결·목표 생성·preview를 확인했습니다. 두 프로토콜 모두 Task A 100,352-step pilot의 held-out 성공이 0/30이었습니다. 원인을 분리하기 위해 현재 기본값은 원본 Task A와 성공 기준을 먼저 재현하는 **프로토콜 v3**입니다. 아직 v3 실제 시뮬레이터 결과와 A/B/C 전체 PPO-EWC 비교 결과는 없습니다.

## 원본 작업공간 유지

- 원본: [ymk2001/Barista_robot_Reinforcement_Learning](https://github.com/ymk2001/Barista_robot_Reinforcement_Learning)
- 참조 commit: `232dd870ff50f988b1168d7630998438964f5423`
- 장면: `safety_rl_2dof.ttt`
- 장면 Git blob SHA-1: `925ea5fcc4c2ecb8cb93c1d8cc11e68c434c7b53`

원본 GitHub에 쓰는 코드는 없습니다. 모든 새 CL 코드는 이 저장소에 있습니다. 장면은 기존 로컬 파일을 지정하거나 `fetch-scene`으로 고정 commit에서 다운로드합니다. `.ttt`를 재작성하거나 저장하지 않으며, 원본 학습 코드·로그·모델을 복사하지 않습니다.

벽, 머신, 로봇 베이스, 카메라 및 기존 Target 오브젝트 위치를 이동하지 않습니다. v3의 Task A는 원본 `/FirstTarget`, `/EndTarget`의 실제 위치를 그대로 사용합니다. B/C는 안전 연결영역 내부에서 선택하지만 Target 오브젝트를 이동하지 않고 저장 좌표로 판정합니다. A/B/C 목표는 모두 **정책에 숫자로 입력**됩니다.

**명령 실행 시 지정한 장면을 CoppeliaSim에 다시 불러옵니다.** 전용 시뮬레이터 인스턴스를 사용하고, 다른 작업 중인 장면은 먼저 저장하세요. 기본적으로 CoppeliaSim과 Python을 동일 PC에서 실행합니다. 원격 서버 사용 시 `--scene` 절대경로가 서버에서도 동일하게 존재해야 합니다.

## 실험 설정

| 항목 | 구현 |
|---|---|
| 작업공간 | 원본 2-DoF 바리스타 장면 |
| Task | A=원본 marker 위치쌍, B/C=중앙의 잘 연결된 안전 위치쌍 |
| 관측(v3 기본) | 84×84 grayscale 영상 4프레임(CHW) + 목표 좌표 6개 |
| 목표 좌표 | 베이스 위치에 대한 병진 오프셋, world 축 방향 |
| 행동 | 두 관절의 속도, 각각 최대 절댓값 0.8 rad/s |
| 보상 | 두 목표까지 거리 감소 합 ×10, 성공 +100, 충돌 −20 |
| 성공 | joint2 및 끝단이 각각 해당 목표 허용 오차 안에 있음 |
| 충돌 | link1/link2/끝단과 wall/machine1/machine2, 비인접 로봇 부품 사이 충돌 |
| 종료 | 충돌 또는 성공 / 최대 step에서 timeout |
| 비교 | 동일 PPO·환경에 EWC 정규화만 추가 |

목표 A/B/C는 **위치쌍 도달 과제**입니다. 컵 파지, 음료 제조, 액체, 독립적인 6D pose 제어는 구현하지 않습니다. 자동 목표에 임의의 픽업/추출/제공 의미를 붙이지 않습니다. 충돌률은 측정하지만 안전 보장이나 PPO-Lagrangian을 구현한 것은 아닙니다.

원본 대비 목표 입력, task 전환, 안전 무작위 초기화, 난수 처리, 프레임 초기화와 평가 체계가 추가됩니다. 따라서 완전히 동일한 원본 재현이 아니라 **CL을 위한 원본 근접 기준선**입니다. 학습은 매 episode마다 연결된 안전 관절공간에서 새로운 연속 자세를 생성합니다. 평가는 별도 seed로 생성·저장한 다양한 held-out 자세를 모든 방법과 seed가 공유합니다. v3 기본 성공 허용 오차 0.15 m는 원본과 같습니다.

v2는 자동으로 가장 멀리 떨어진 목표와 0.05 m 기준을 사용했고, v2.1은 여기에 관절각과 시작점 curriculum을 추가했습니다. 이 변화들이 동시에 섞이면 실패 원인을 알기 어렵습니다. v3에서는 먼저 원본 Task A, 0.15 m, task별 `3e-4 → 5e-5` 선형 학습률 감소, 전체 안전 시작분포를 사용합니다. 관절각과 curriculum은 삭제하지 않고 설정 기반 ablation으로 남겨 둡니다.

## 1. 설치와 사양 확인

권장 시작 환경: Python 3.10, 원본 README 기준 CoppeliaSim 4.9.0 Edu.
SB3 2.7.0 / Gymnasium 1.2.0으로 고정했습니다. GPU는 필수가 아니며 CPU도 지원합니다.

```bash
git clone https://github.com/AriaKang0101/Barista_robot_Continual_Learning.git
cd Barista_robot_Continual_Learning
conda create -n barista-cl python=3.10 -y
conda activate barista-cl
```

CPU로 먼저 시작하는 예:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m barista_cl doctor
python -m pytest -q
```

GPU를 쓰려면 드라이버에 맞는 PyTorch 2.6 빌드를 [공식 설치 안내](https://pytorch.org/get-started/previous-versions/)에 따라 먼저 설치하세요. 요구사항은 torch 2.3~2.6을 허용하지만 로컬 검증은 2.6.0에서 했습니다. 기존 연구 환경을 덮어쓰지 않도록 별도 conda 환경을 권합니다.

`doctor`는 Python·패키지·CUDA·GPU 정보를 출력합니다. 시뮬레이터에 접속하거나 장면을 변경하지 않습니다. 문제가 있으면 이 출력과 오류를 공유하세요. 사전에 GPU 모델을 지정할 필요는 없습니다.

## 2. 동일 장면 준비

```bash
python -m barista_cl fetch-scene
```

`scenes/safety_rl_2dof.ttt`에 원본을 다운로드하고 해시를 검증합니다. 기존 파일이 다르면 덮어쓰지 않습니다. 이미 가진 원본 파일을 쓰려면 이후 명령의 `--scene`에 경로를 전달하세요.

```bash
python -m barista_cl prepare --scene /absolute/path/safety_rl_2dof.ttt --output artifacts/tasks_v3.json
```

CoppeliaSim GUI를 실행해 두세요. 기본 ZeroMQ 포트는 23000입니다. 명령들이 장면 로드·시작·종료·step을 제어하므로 동시에 GUI에서 조작하지 마세요. 파일 해시가 다르면 원본과 동일한 환경 조건을 위해 실행을 중단합니다.

CoppeliaSim 물리 엔진을 시작할 때 고정 형상이 1 mm 미만으로 안착할 수 있습니다. 코드의 고정 작업공간 검사는 이 수치 오차를 고려해 최대 matrix element 변화 2 mm까지만 허용하며, 그보다 큰 위치·자세 변화는 중단하고 변화량을 출력합니다. 이 허용치는 기본 충돌 간격 1 cm 및 목표 허용 오차 15 cm보다 작습니다.

## 3. 목표 A/B/C 자동 생성

```bash
python -m barista_cl prepare --output artifacts/tasks_v3.json
```

실제 장면에서 다음을 수행합니다.

1. 원본 관절 한계와 탐색 범위(joint1 −110~110°, joint2 −90~90°)의 교집합을 샘플링합니다.
2. 링크와 벽/머신의 충돌 및 기본 1 cm 간격을 검사합니다.
3. 인접 자세 사이를 최대 2° 간격으로 검사해 연결 그래프를 만듭니다.
4. Task A는 원본 `/FirstTarget`, `/EndTarget` 좌표를 사용합니다. 거친 grid는 연결성에만 사용하고, 유망한 grid 사이를 연속 관절각으로 세분 탐색해 실제 성공영역 안의 충돌 없는 자세를 경로 종점으로 찾습니다. 원본 PPO와 같이 목표 종점은 충돌 여부로 판정하며 1 cm 여유는 요구하지 않습니다.
5. B/C는 가장 큰 연결 영역의 중앙에 가깝고 graph 연결도가 높은 자세 중 서로 성공영역이 겹치지 않게 선택합니다. 좌표를 독립적으로 임의 생성하지 않습니다.
6. 가장 큰 안전 연결영역 전체를 학습 난수 sampler의 anchor로 저장합니다. 학습 reset마다 anchor 주변의 새로운 연속 관절각을 만들고 충돌·1 cm 거리·비인접 자기충돌·목표 중복을 검사합니다. 목표 종점 조건과 달리 무작위 시작 자세의 1 cm 안전 여유는 계속 유지합니다.
7. 기본 30개의 서로 다른 평가 자세를 별도 난수로 생성합니다. 각 자세는 안전 연결영역에 이어져야 하며 A/B/C 모두에 대해 실제 물리 step 경로검증을 통과해야 합니다.
8. 물리 시작 후에도 충돌·안전거리를 다시 검사합니다. 모든 조건을 통과한 경우에만 schema v3 task 파일을 저장합니다.

Remote API 호출이 많아 시간이 걸릴 수 있으며 진행 상황이 출력됩니다. 기존 v2/v2.1 파일과 결과는 보존하고 새 `artifacts/tasks_v3.json`을 만드세요.

```bash
python -m barista_cl prepare --grid-size 21 --tolerance 0.15 --eval-starts 50 --output artifacts/tasks_v3_50eval.json
```

실패하면 유효한 목표를 꾸며서 채우지 않고 중단합니다. 출력된 경로 timeout·충돌 등을 확인하세요. `--max-steps`, `--clearance`를 바꾸면 프로토콜이 바뀝니다. 두 방법에 같은 준비 파일을 사용해야 합니다.

학습 초기화는 Gymnasium seed로 재현 가능하지만 매 episode 새 연속 자세를 생성하므로 두 방법이 정확히 같은 순서의 자세를 경험한다고 보장하지 않습니다. 대신 동일한 안전 분포에서 학습합니다. 평가는 저장된 held-out 자세를 각각 정확히 한 번 사용하므로 모든 방법·seed의 비교 조건이 같습니다. 검사는 지정된 충돌 쌍과 이산 시간/각도 샘플에 대한 것이며 모든 연속 경로의 절대적 안전을 보장하지 않습니다.

v3 기본값은 curriculum 없이 처음부터 전체 안전 연결영역에서 학습하며 관절각을 정책에 입력하지 않습니다. `observation_mode=image_goal_joints`와 v2.1의 curriculum 값은 이후 ablation에서만 사용하세요. 어떤 경우에도 최종 평가는 저장된 held-out 자세 전체를 사용합니다.

검증 제어기의 움직임 확인:

```bash
python -m barista_cl preview --task A
python -m barista_cl preview --task B
python -m barista_cl preview --task C
```

**preview는 경로 검증 제어기이며 학습된 PPO가 아닙니다.** 경로를 PPO의 행동 라벨이나 시연 데이터로 사용하지 않습니다.

## 4. 단일 Task 예비 학습

먼저 1,024 step smoke test로 실행 연결만 확인한 뒤, 원본 Task A 전체 pilot을 실행합니다.

```bash
python -m barista_cl train --method sequential --only-task A --config configs/smoke_v3.json --output runs/v3_smoke_A
python -m barista_cl train --method sequential --only-task A --output runs/v3_pilot_A
```

Task A가 held-out 시작점에서도 학습된 것을 확인한 뒤 B/C를 각각 새 모델로 확인합니다.

```bash
python -m barista_cl train --method sequential --only-task B --output runs/v3_pilot_B
python -m barista_cl train --method sequential --only-task C --output runs/v3_pilot_C
```

처음부터 배우지 못한 작업의 낮은 성능을 ‘망각’으로 해석하면 안 됩니다. 준비 제어기의 성공은 물리적 도달 가능성 검사이며 PPO의 학습 가능성과 별도입니다.

## 5. 순차 학습 비교

동일 seed로 첫 비교:

```bash
python -m barista_cl train --method sequential --seed 0 --output runs/sequential_seed0
python -m barista_cl train --method ewc --seed 0 --output runs/ewc_seed0
python -m barista_cl compare runs/sequential_seed0 runs/ewc_seed0 --output artifacts/comparison_seed0
```

3개 seed를 자동 실행:

```bash
python scripts/run_comparison.py --seeds 0 1 2 --output runs/comparison
```

하나의 CoppeliaSim 서버에서는 **직렬 실행**합니다. 학습 단계가 끝난 후 같은 장면에서 평가하고, 다음 학습 전에 관측과 프레임을 초기화합니다. 서로 다른 프로세스로 동시에 학습/평가하지 마세요.

| 완료 단계 | 학습 데이터 | 동결 정책 평가 |
|---|---|---|
| 1 | A만 | A |
| 2 | B만 | A, B |
| 3 | C만 | A, B, C |

Task 전환 시 모델과 optimizer를 새로 생성하지 않습니다. Task 경계는 알려져 있고 목표 좌표는 항상 정책에 주어집니다. 이전 영상/transition을 재학습하지 않습니다. 이전 환경은 평가용으로만 실행합니다.

기본값은 `configs/default.json`에서 수정합니다.

| 설정 | 기본값 | 의미 |
|---|---:|---|
| timesteps_per_task | 100352 | 512의 배수인 정확한 rollout 예산 |
| n_steps / batch_size / n_epochs | 512 / 128 / 10 | PPO 업데이트 |
| learning_rate_start / end | 0.0003 / 0.00005 | 각 task에서 다시 시작하는 선형 감소 |
| ewc_lambda | 1000 | 탐색 시작값이며 최적값 아님 |
| fisher_samples | 128 | 마지막 현재-task rollout의 상태 표본 수 |
| curriculum_initial_fraction | 1.0 | 처음부터 전체 안전 anchor 사용 |
| curriculum_full_fraction | 1.0 | curriculum 비활성화 |
| observation_mode | image_goal | 관절각을 제외한 v3 기준 관측 |
| task_order | A, B, C | 작업 순서 |

lambda는 Fisher 크기와 네트워크에 의존합니다. 10/100/1000 같은 후보를 별도 예비 검증에서 비교하고, 최종 평가 초기 자세로 값을 고르지 마세요. 망각이 거의 발생하지 않는 것도 결과입니다. EWC 효과를 보이기 위해 불리한 작업 순서만 선택하지 마세요.

## EWC 구현

유효한 목적함수는 `L = L_PPO + (lambda/2) × Σ_과거작업 Σ_파라미터 F × (현재값 − 과거값)²`입니다.

- 작업별 정책 스냅샷과 Fisher를 누적합니다. C 학습에서는 A와 B가 보호됩니다.
- 중요도는 정책 `log π(a|o,g)`의 **샘플별 gradient 제곱을 평균**한 값입니다. 평균 gradient를 제곱하지 않습니다.
- 상태는 예산에 포함된 마지막 현재-task rollout에서 가져옵니다. 행동은 consolidation 시점 정책에서 새로 샘플링하고 detach합니다. likelihood에는 clipping 전 행동을 사용합니다.
- 이는 마지막 rollout의 경험적 상태분포에 대한 Fisher 근사입니다. 최종 정책으로 추가 rollout을 수집한 정확한 on-policy 상태분포 추정은 아닙니다. 추가 시뮬레이터 예산을 사용하지 않는 명시적인 근사입니다.
- 정책과 연결된 공유 특징 추출부·actor를 보호합니다. 정책 likelihood gradient가 없는 value-only head는 보호하지 않습니다.
- SB3 backward 중 gradient hook으로 EWC 미분을 더합니다. **global gradient clipping 이전에** 합쳐집니다. 명시적인 penalty 미분과의 동치성을 테스트합니다. SB3 PPO 업데이트 루프 자체는 복사하거나 교체하지 않습니다.
- TensorBoard `train/loss`는 SB3 원래 loss입니다. 추가 항은 `ewc/penalty_after_update`에 별도로 기록됩니다.
- lambda=0에서 기본 SB3 PPO와 정확히 같은 업데이트를 수행하는지 테스트합니다.
- Fisher/평가는 Python·NumPy·PyTorch 난수 상태를 보존합니다.
- 체크포인트는 정책·optimizer·이전 파라미터·Fisher를 포함합니다. 자동 run 재개 CLI는 제공하지 않습니다. 실패한 실험은 별도 출력 디렉터리로 다시 실행하세요.

## 결과와 지표

| 파일 | 내용 |
|---|---|
| config.json / tasks.json | 실제 설정, 목표/초기 자세, 물리 검증 기록 |
| manifest.json | seed, 버전, 장면 설정 식별값, 학습 step, 완료/실패 상태 |
| stage_1_A.zip 등 | 단계별 정책·optimizer·EWC |
| training_episodes.csv | 학습 보상·성공·충돌·길이 |
| evaluation_episodes.csv | 성공·충돌·timeout·최종 오차·step |
| success_matrix.csv | 행=완료 단계, 열=평가 Task, 아직 안 배운 작업은 NaN |
| metrics.json | 최종 평균 성공률·평균 망각량·BWT |
| tensorboard/ | PPO/EWC 로그 |

`compare`는 task 파일·설정·실행 환경·학습 예산과 seed 집합이 일치하는 완료 run만 비교합니다. `retention.png`, `per_seed_metrics.csv`, `summary.json`을 생성합니다. 그래프 띠는 **훈련 seed 간 표준편차**이며 신뢰구간이 아닙니다. seed 하나로는 표준편차를 제시하지 않습니다.

`R[i,j]`를 i단계 학습 후 j작업 성공률이라고 할 때:

- 최종 평균 성공률: 마지막 행의 평균.
- 평균 망각량: 이전 작업 각각의 `이전에 학습한 이후 최고 성공률 − 최종 성공률`의 평균.
- BWT: 이전 작업 각각의 `최종 성공률 − 처음 그 작업을 배운 직후 성공률`의 평균.

성공률은 0~1입니다. 0.4 감소는 40%p 감소입니다. 망각량은 최종 성능이 이전 최고보다 좋아지면 음수일 수 있습니다. BWT 양수는 이전 작업 성능 향상입니다. 충돌이 적어도 정지해서 timeout만 발생할 수 있으므로 **성공률·충돌률·timeout을 함께** 해석하세요.

```bash
tensorboard --logdir runs
```

## 코드 구조

| 경로 | 역할 |
|---|---|
| barista_cl/simulator.py | 원본 장면 검증, ZeroMQ, 관절·충돌·영상 |
| barista_cl/prepare.py | 기구학 기반 목표·초기 자세·경로 검증 |
| barista_cl/env.py | 두 방법의 공통 goal-conditioned 환경 |
| barista_cl/ewc.py | policy Fisher, 중요도 누적, PPO-EWC |
| barista_cl/experiment.py | 학습·동결 평가·체크포인트·CSV |
| barista_cl/report.py | 조건 검사·지표·그래프 |
| barista_cl/core.py | 설정 검사·경로 그래프·CL 지표 |
| barista_cl/__main__.py | CLI·사양 확인·장면 다운로드 |
| scripts/run_comparison.py | 여러 seed 직렬 실행 |
| tests/ | 알고리즘·환경·전체 실행 흐름 테스트 |

## 검증 범위와 문제 해결

로컬 테스트 환경: Python 3.12, PyTorch 2.6.0, SB3 2.7.0, Gymnasium 1.2.0, CPU. GitHub Actions는 Python 3.10 CPU 테스트로 구성했습니다. Actions 통과 여부는 실제 실행 결과를 확인하세요.

테스트는 lambda=0 PPO 일치, Gaussian Fisher sanity check, EWC gradient 동치성, 중요도 누적, 체크포인트 재로딩, RNG 보존, 충돌 우선 처리, 목표 생성 실패 차단, 모의 영상 환경에서 두 방법×3단계 실행 및 그래프 생성을 포함합니다.

**개발 환경에서는 CoppeliaSim v3 준비·학습을 직접 실행하지 못했습니다.** 실제 장면에서 먼저 `prepare → preview A/B/C → v3 smoke A → v3 full pilot A` 순서로 검증해야 합니다. Task A가 학습되지 않으면 B/C 또는 CL 비교로 넘어가지 마세요.

- 연결 timeout: CoppeliaSim 실행, ZeroMQ 포트 23000, host/port를 확인하세요.
- 해시 불일치: 저장 후 변경된 장면일 수 있습니다. 원본 사본을 별도 경로에 받으세요.
- 객체 경로 오류: 원본 계층 이름이 달라졌는지 확인하세요.
- 연결 영역 부족: 출력과 장면을 확인한 뒤 grid/clearance를 조정하세요.
- 동적 검증 실패: 준비 제어기가 경로를 수행하지 못했습니다. `all_passed`를 수동 변경하지 마세요.
- 영상 없음: Vision Sensor·렌더링 설정을 확인하세요. 처음부터 headless로 실행하지 않는 것을 권합니다.
- 출력 폴더 존재: 기존 결과 보존을 위해 새 `--output`을 사용하세요.
- GPU 문제: `--device cpu`로 검사하고 `doctor` 출력을 확인하세요.

## 참고

- [EWC 원 논문](https://arxiv.org/abs/1612.00796)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html)
- [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/)
- [CoppeliaSim ZeroMQ API](https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm)

일반 EWC 비교를 위한 연구 시작점입니다. 실제 결과 확인 후 안전 비용 보존, 작업 순서·seed 확대, 더 다양한 초기 자세 등으로 확장할 수 있습니다.

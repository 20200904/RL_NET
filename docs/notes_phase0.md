# Phase 0 노트 — 환경 확인

## 무엇을 만들었나

- `src/envs/compat.py`: NASim 환경 생성 헬퍼와 `IntAction` wrapper.
- `src/phase0/check_env.py` (`python -m src.cli check-env`): tiny/small/gen6/gen8을 로드해 무작위 정책 1000스텝을 돌리고 초당 스텝 수를 `results/phase0/check_env.csv`에 기록.
- `src/phase0/sb3_check.py` (`python -m src.cli sb3-check`): SB3 PPO 기준선. run마다 `episodes.csv`와 `done.json`을 남기고, `done.json`이 있으면 건너뛴다(재개 가능).

## 설치와 호환성

- PC에 Python이 없어서 `uv`로 Python 3.11.17 venv를 만들었다. git은 설치되어 있지 않다.
- nasim 0.12.0은 numpy 2.4.6, gymnasium 1.4.0과 설치 충돌 없이 동작한다. 버전을 낮출 필요가 없었다.
- 문제 1건: NASim은 flat action에 대해 `isinstance(action, int)`를 assert한다(`nasim/envs/action.py`, `FlatActionSpace.get_action`). `np.int64`는 거부되므로 `IntAction` wrapper에서 `int(action)`으로 바꾼다. NASim은 수정하지 않았다.

## 6.1 비용 판정에 필요한 속성 (소스에서 직접 확인)

| 필요한 정보 | 접근 방법 | 파일 |
|---|---|---|
| 정수 행동 → Action 객체 | `env.action_space.get_action(int(a))` | `envs/action.py` `FlatActionSpace` |
| 대상 호스트 주소 | `action.target` → `(subnet, host)` | `envs/action.py` `Action` |
| 행동 종류 | `action.is_exploit()`, `is_privilege_escalation()`, `is_scan()`, `is_service_scan()`, `is_os_scan()`, `is_subnet_scan()`, `is_process_scan()` | 같은 파일 |
| exploit 정의 | `action.name`, `action.service`, `action.os`(None이면 OS 무관) | `Exploit` |
| 호스트의 OS·서비스(정적) | `env.scenario.hosts[addr].os`, `.services` (이름→bool dict) | `scenarios/host.py` |
| 호스트의 현재 상태 | `env.current_state.get_host(addr)` → `.is_running_os(name)`, `.is_running_service(name)`, `.compromised`, `.reachable`, `.discovered`, `.access` | `envs/state.py`, `envs/host_vector.py` |
| 목표 호스트 | `env.scenario.sensitive_addresses` | `scenarios/scenario.py` |
| 행동 결과 | `info`의 `success`, `connection_error`, `permission_error`, `undefined_error` | `ActionResult.info()` |

결론: C1, C2 모두 wrapper만으로 판정할 수 있다. (wrapper를 실제로 구현해 검증하는 것은 Phase 1.)

## 설계에 영향을 주는 발견

1. **토폴로지는 시드로 변하지 않는다.** `ScenarioGenerator._generate_subnets/_generate_topology`는 호스트 수만으로 결정된다. 6호스트면 항상 DMZ 1, sensitive 1, user 4. 시드로 바뀌는 것은 호스트의 OS·서비스·프로세스, exploit/privesc 정의, 방화벽뿐이다. 목표 호스트도 기본값은 고정 위치(`random_goal=True`로만 user 쪽 하나가 바뀐다).
2. **exploit의 의미가 네트워크마다 달라진다.** exploit 정의(대상 서비스·OS·획득 권한)가 시드마다 새로 뽑히므로, 같은 행동 번호가 네트워크마다 다른 exploit이다. 관측에는 exploit 정의가 없어서 에이전트가 이를 알 방법이 없다. 일반화 실험(RQ3)을 하려면 exploit 표를 전 네트워크 공통으로 고정해야 한다.
3. **부분 관측에서는 관측에 기억이 없다.** `State.get_observation`은 매 스텝 빈 관측에 "방금 행동의 결과"만 채운다. 스캔 결과가 다음 스텝에 사라지므로, 기억 없는 MLP는 "스캔 결과를 보고 fragile 여부를 추론"할 수 없다. 스캔 결과를 에피소드 동안 누적하는 관측 wrapper가 필요하다.
4. **`HostVector`가 클래스 변수에 인덱스 맵을 둔다.** 한 프로세스에 모양이 다른 시나리오(예: tiny와 gen6)를 동시에 띄우면 깨진다. 모양이 같은 생성 네트워크끼리는 안전하다.
5. **NASim은 전역 `np.random`을 쓴다.** 재현성을 위해 exploit 성공 확률을 1.0으로 두거나 전역 시드를 관리해야 한다.
6. **생성기 기본값은 스텝 한도가 없다.** `step_limit`을 반드시 넘겨야 한다(현재 400).

## 측정에서 알게 된 것

- 무작위 정책도 스텝 한도 안에 목표에 도달한다. "목표 도달률"만으로는 학습 여부를 판단할 수 없고, 에피소드 길이나 결정적 평가를 함께 봐야 한다. `sb3_check.py`의 `steps_to_solve`(도달률 기준)는 이 때문에 의미가 약하다.
- gen6(net_seed=1)은 학습 후 4스텝에 풀린다. 과제가 너무 쉬울 수 있다.

수치는 `docs/compute_budget.md` 참고.

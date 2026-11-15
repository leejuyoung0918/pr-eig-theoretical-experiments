# 실험 코드를 재사용 가능한 연구 도구로 만들기

## 해결하는 문제

선호 학습의 query rule을 비교할 때, 한 지표만 보거나 실행 환경이 달라지면 정보량·수렴 속도·계산 비용의 차이를 잘못 해석할 수 있습니다. 이 저장소는 controlled·sequential·randomized 실험을 분리하고, seed·checkpoint·metadata·보고서 생성 절차를 코드로 남깁니다.

## 결과와 비판적 기술 선택

보존된 실험은 PR-EIG, Exact-RC, Fisher-RC를 같은 finite-noise MNL 설정에서 비교합니다. 저장된 uncached benchmark의 중앙 설계 시간은 각각 3.958 ms, 4.789 ms, 0.291 ms입니다. 이 수치는 당시 실험 환경의 결과이며 일반적인 성능 보장은 아닙니다.

Fisher-RC의 비용이 낮다는 사실만으로 모든 설정에서 우월하다고 결론 내리지 않습니다. `rho*=0.25` 결과에는 calibrated method의 query-count 이점이 없고, 더 미세한 정밀도에서는 coverage도 함께 확인합니다. 근거는 [최종 보고서](../outputs/experiment1/full/FINAL_REPORT.md)입니다.

## 개발 환경 개선

이번 정리에서 다음의 설치·검증 문제를 수정했습니다.

| 이전 문제 | 변경 | 검증 방식 |
|---|---|---|
| `experiment1`이 wheel의 명시적 package 목록에서 누락 | `pyproject.toml`에 포함 | 실제 wheel을 설치해 소스 checkout 밖에서 전체 테스트와 CLI 도움말 실행 |
| clone 이름을 `figure2_experiments`로 맞추고 부모 폴더에서 실행해야 한다는 안내 | editable install을 표준 개발 절차로 사용 | 기본 clone 이름에서도 package import |
| dependency 선언을 두 곳에서 관리 | requirements는 pyproject의 test extra 설치로 연결 | 한 곳에서 런타임·테스트 의존성 관리 |
| 로컬 테스트만으로 설치 결과를 보장할 수 없음 | Python 3.10·3.12 CI에서 build·lint·wheel 테스트 | 배포 누락을 로컬 경로가 가리지 않도록 검사 |

## 배운 점

- **재현 가능한 실험과 설치 가능한 도구는 다른 계약**입니다. editable checkout에서 import가 성공해도 wheel에서 빠질 수 있습니다.
- 계산 비용·정확도·정밀도별 coverage를 함께 봐야 새로운 query rule의 도입 조건을 정할 수 있습니다.
- 재실행 비용이 큰 연구에서는 deterministic checkpoint와 결과에서 보고서를 다시 만드는 경로가 개발 생산성의 일부입니다.

## 개발 가이드

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ruff check .
python -m pytest
python -m build
```

원본·수정 모델의 수치 동작은 기존 회귀 테스트로 확인합니다. 모든 연구 실험을 CI에서 다시 돌리지는 않습니다. 기존 결과를 활용하는 실험·그림 재생은 checkout의 editable 설치를 사용하세요. wheel은 코드만 포함하며 보존된 결과와 대용량 원자료를 배포하지 않습니다. 기본 실험 출력은 package의 `outputs/` 아래에 생성되므로 연구 실행에는 writable checkout을 권장합니다.

참고: [setuptools package 명시](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html), [pytest 설치 환경 검사](https://docs.pytest.org/en/stable/explanation/goodpractices.html).

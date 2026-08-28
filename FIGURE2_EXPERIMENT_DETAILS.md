# Figure 2 시뮬레이션 실험 상세 기술서 (Experiment Details)

본 문서는 NeurIPS 논문의 **Figure 2**(및 관련 부록 Figure) 생성을 위해 수행된 시뮬레이션 및 벤치마크 실험의 이론적 배경, 수학적 모델, 알고리즘, 파라미터 설정, 실험 파이프라인, 통계 분석 기법, 결과 데이터를 종합적이고 매우 상세하게 정리한 기술 보고서입니다.

---

## 1. 실험의 이론적 배경 및 목적 (Theoretical Context & Objectives)

### 1.1 연구 문제: MNL 모델 하에서의 능동 선호 학습 (Active Preference Learning)
- **목표**: 잠재 공간(Latent space) 상의 타겟 위치(또는 사용자 선호 파라미터) $\theta \in \mathbb{R}^d$를 $M$-지선택 쿼리(Slate Queries)를 통해 최소한의 질의 수로 정밀하게 추정하는 것입니다.
- **쿼리 형태**: 현재 사후분포 중심 $x$와 주축 방향 $u$에 대해, 반경 $r$과 정규화된 템플릿 $b = (b_1, \dots, b_M) \in \mathbb{R}^M$를 배치하여 $M$개의 선택지를 제시합니다:
  $$x_m = x + r b_m u \quad (m = 1, \dots, M)$$
- **반응 모델 (Squared-distance MNL Likelihood)**:
  타겟과 쿼리 중심 사이의 1차원 투영 오프셋을 $s = (\theta - x)^\top u$라 할 때, 사용자가 $m$번째 선택지를 고를 확률은 다음과 같습니다:
  $$P(Y = m \mid s, r) = \frac{\exp\left[-\beta (s - r b_m)^2\right]}{\sum_{j=1}^M \exp\left[-\beta (s - r b_j)^2\right]}$$
  여기서 $\beta$는 역온도(inverse temperature, 노이즈 수준의 역수)입니다.

### 1.2 템플릿 정규화 (Template Normalization)
$M$개 선택지의 템플릿 $b$는 표준정규분포 $\Phi$의 등질량 분위수(Quantile) 중심에서 생성되며, 엄격하게 평균 0, RMS 1로 정규화됩니다:
1. 원시 분위수 중심: $b_m^{\text{raw}} = \Phi^{-1}\left(\frac{m + 0.5}{M}\right) \quad (m = 0, \dots, M-1)$
2. 중심화: $b \leftarrow b - \text{mean}(b)$
3. 단위 RMS 정규화: $b \leftarrow \frac{b}{\sqrt{\text{mean}(b^2)}}$

### 1.3 핵심 문제의식: 1단계 정보 붕괴 (Information Collapse)
- 사후 불확실성(표준편차 $\sigma$)이 작아질 때($\sigma \to 0$), 쿼리 반경 $r$을 어떻게 결정해야 하는가?
- **Posterior-Relative EIG (PR-EIG)**:
  - 사후분포 스케일에 비례하게 쿼리 반경을 축소 ($r = \alpha \sigma$, $\alpha \in [0.05, 20]$).
  - 그러나 유한한 노이즈($\beta < \infty$) 환경에서는 $\sigma \to 0$일 때 1단계 상호정보량(EIG)이 $\mathcal{O}(\sigma^4)$로 급격히 소멸(Collapse)합니다.
- **Radius Calibration (RC)**:
  - **Exact-RC**: 사후 불확실성에 묶이지 않고 물리적 반경 $r$을 직접 EIG 최대화로 최적화.
  - **Fisher-RC**: $s=0$에서의 스칼라 Fisher 정보량 $I_F(r)$을 최대화하는 물리적 반경 $r^*$을 결정 ($I_F(r^*) = \max_r I_F(r)$).
  - 두 RC 기법 모두 $\sigma \to 0$ 극한에서 1단계 정보량 $\mathcal{O}(\sigma^2)$를 유지합니다.

---

## 2. 전체 실험 구조 및 워크플로우 (Experimental Architecture)

Figure 2는 다음 3개의 핵심 패널로 구성되어 이론 증명에서 다차원 실증 벤치마크까지 단계적으로 검증합니다:

```
+-----------------------------------------------------------------------------------+
|                                 Figure 2 전체 구조                                |
+-----------------------------------------------------------------------------------+
| [Panel (a) Controlled Theory]                                                     |
| 1D Gauss-Hermite 구적법 (정밀도 검증) -> 1단계 정보량 I(sigma) 스케일링             |
| PR-EIG: O(sigma^4) vs Exact/Fisher-RC: O(sigma^2) 확인                            |
+-----------------------------------------------------------------------------------+
| [Panel (b) Sequential Consequence]                                                |
| 1D Gaussian ADF 순차 탐색 (200 trials, N=100,000 쿼리)                             |
| 분산 감쇄율: PR-EIG: O(n^{-1/4}) vs Exact/Fisher-RC: O(n^{-1/2}) 확인             |
+-----------------------------------------------------------------------------------+
| [Panel (c) Randomized Multidimensional Robustness]                                |
| 500개 무작위 환경 (d=2~16, M=3~9, beta=0.25~4, kappa=1~100)                      |
| Projected Particle ADF + Precision Sweep (rho* = 0.50 -> 0.02)                   |
| 쿼리 수 비 (Fisher/PR, Exact/PR)의 고정밀도 영역 이득 실증                         |
+-----------------------------------------------------------------------------------+
```

---

## 3. 세부 실험 내용 및 방법론 (Detailed Methodology by Panel)

---

### 3.1 Panel (a): 통제된 1단계 정보 스케일링 실험 (Controlled Information Scaling)

#### 1) 목적
$\sigma \to 0$일 때 1단계 기대 정보 획득량(EIG, $I(\sigma)$)이 이론적 예측($\sigma^4$ vs $\sigma^2$)과 일치하는지 결정론적 고정밀도 수치적분을 통해 검증합니다.

#### 2) 세부 파라미터 및 환경
- $\beta = 1.0$, $M = 5$
- 사후 표준편차 범위: $\sigma \in [10^{-4}, 1.0]$ (로그 스케일 121개 지점)
- 수치적분: 128차 Gauss-Hermite Quadrature (256차 구적법과 비교하여 최대 상대오차 $1.15 \times 10^{-7}$ 이하 보증)
- 탐색 범위:
  - PR-EIG: $\alpha \in [0.05, 20.0]$ (81~241 그리드 점 + Bounded Scalar Refinement)
  - Exact-RC: 물리적 반경 $r \in [10^{-4}, 10.0]$
  - Fisher-RC: $s=0$에서 Fisher 정보량 최대화 반경 $r$ 선택 후, 해당 $r$에서의 EIG를 128차 구적으로 계산

#### 3) 수치적 안정성 기법
확률 $p$와 주변확률 $q$가 거의 같을 때 float64 연산에서 발생하는 1차 소거 오차를 방지하기 위해 KL Divergence를 다음과 같이 변형하여 계산:
$$x = \frac{p - q}{q}, \quad \text{KL}(p \parallel q) = \sum q \cdot \left[(1+x)\ln(1+x) - x\right]$$

#### 4) 통계 피팅 및 결과 (Fit Range: $\sigma \in [10^{-4}, 10^{-2}]$)
| 정책 (Policy) | 이론적 기울기 ($\text{Slope}_{\text{theory}}$) | 실측 기울기 ($\text{Fitted Slope}$) | 표준오차 (SE) | 95% 신뢰구간 (CI) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PR-EIG** | **4.0** | **3.995858** | $5.36 \times 10^{-4}$ | $[3.994786, 3.996931]$ | 0.999999 |
| **Exact-RC** | **2.0** | **1.999991** | $1.18 \times 10^{-6}$ | $[1.999988, 1.999993]$ | 1.000000 |
| **Fisher-RC** | **2.0** | **1.999991** | $1.18 \times 10^{-6}$ | $[1.999988, 1.999993]$ | 1.000000 |

> **핵심 결론**: PR-EIG는 사후 스케일에 비례하게 반경을 줄임으로써 $\sigma^4$의 정보 붕괴를 일으키는 반면, Exact-RC 및 Fisher-RC는 물리적 반경을 적절히 유지하여 $\sigma^2$ 정보량을 확보함을 완벽히 입증했습니다.

---

### 3.2 Panel (b): 장기 순차적 탐색 수렴 속도 (Long-Horizon Sequential Convergence)

#### 1) 목적
1단계 정보 획득률($\sigma^4$ vs $\sigma^2$)의 차이가 장기 순차적 능동 탐색 과정($N = 100,000$ 쿼리)에서 사후 불확실성 $\epsilon_n = \text{RMS}(\sigma_n)$의 감쇄 속도에 미치는 영향을 검증합니다.

#### 2) 이론적 유도
- 불확실성 감쇄 미분방정식: $\frac{d\sigma^2}{dn} \approx - I(\sigma)$
- **PR-EIG ($I(\sigma) \propto \sigma^4$)**:
  $$\frac{d\sigma^2}{dn} \propto -\sigma^4 \implies \frac{1}{\sigma^2} \propto n \implies \sigma(n) \propto n^{-1/4} \quad (\text{Slope} = -0.25)$$
- **Exact-RC / Fisher-RC ($I(\sigma) \propto \sigma^2$)**:
  $$\frac{d\sigma^2}{dn} \propto -\sigma^2 \implies \ln \sigma^2 \propto -n \implies \text{Fisher asymptotic regime: } \sigma(n) \propto n^{-1/2} \quad (\text{Slope} = -0.5)$$

#### 3) 세부 파라미터 및 알고리즘
- **사후분포 갱신**: Gaussian Assumed-Density Filtering (ADF)
  - 매 쿼리마다 실제 MNL 우도를 128차 Gauss-Hermite 구적으로 수치적분하여 1차/2차 모멘트(평균 이동, 분산 비)를 정확히 계산 후 가우시안으로 모멘트 매칭.
- **반복 수 및 쿼리 수**: 200회 독립 시행(Repetitions), $N_{\max} = 100,000$ 쿼리
- **공통 난수 (Common Random Numbers, CRN)**: 모든 정책이 동일한 초기 타겟 draw와 동일한 라운드별 균등난수($U_n \sim \text{Uniform}(0,1)$) 스트림을 공유하여 분산을 최소화하고 공정하게 비교.
- **전이 캐시 (Transition Cache)**: $\log \sigma$에 대해 800개 조밀 그리드로 설계 반경 및 응답별 모멘트 전이표를 사전 구축하여 연산 가속.

#### 4) 통계 피팅 및 결과 (Primary Fit Window: $n \in [10,000, 100,000]$, 2,000 Bootstrap Resamples)
| 정책 (Policy) | 이론적 수렴 지수 | 실측 기울기 ($\text{Fitted Slope}$) | 95% 회귀 신뢰구간 | 95% 부트스트랩 신뢰구간 | 최종 RMS $\epsilon_{10^5}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PR-EIG** | **-0.25** | **-0.261509** | $[-0.262401, -0.260617]$ | $[-0.261509, -0.261509]$ | $7.557 \times 10^{-3}$ |
| **Exact-RC** | **-0.50** | **-0.499980** | $[-0.499982, -0.499978]$ | $[-0.499980, -0.499980]$ | $2.318 \times 10^{-3}$ |
| **Fisher-RC** | **-0.50** | **-0.499985** | $[-0.499986, -0.499983]$ | $[-0.499985, -0.499984]$ | $2.318 \times 10^{-3}$ |

> **핵심 결론**: PR-EIG는 장기 수렴에서 $n^{-1/4}$ 수준으로 수렴이 극도로 지연되는 반면, Exact-RC와 Fisher-RC는 점근적으로 최적인 $n^{-1/2} = \mathcal{O}(1/\sqrt{n})$ 속도로 빠르게 타겟을 추정함을 입증했습니다.

---

### 3.3 Panel (c): 다차원 무작위화 강건성 및 정밀도 스윕 (Randomized Multidimensional Robustness)

#### 1) 목적
단순 1차원/고정 파라미터가 아닌, 임의의 차원, 슬레이트 크기, 노이즈, 이방성(Anisotropy)을 가진 500개 무작위 환경에서 물리적 보정(RC)이 실제로 요구 쿼리 수를 얼마나 절감하는지 정밀도 수준($\rho^*$)별로 종합 평가합니다.

#### 2) 무작위 환경 생성 분포 (500 Paired Environments)
각 환경 $e$는 시드 기반 결정론적 생성기로 독립 추출:
- **차원 ($d$)**: $\{2, 4, 8, 16\}$ 중 균등 선택
- **슬레이트 크기 ($M$)**: $\{3, 5, 7, 9\}$ 중 균등 선택
- **역온도 ($\beta$)**: $[0.25, 4.0]$ (Log-uniform 분포)
- **공분산 조건수 ($\kappa$, Anisotropy)**: $[1.0, 100.0]$ (Log-uniform 분포)
- **초기 스케일 ($s_0$)**: $[0.5, 2.0]$ (Log-uniform 분포)
- **공분산 행렬 ($\Sigma_0$)**: 무작위 Haar 직교행렬 $Q$ (QR 분해)와 고유값 $\lambda_i \in [1, \kappa]$를 통해 $\Sigma_0 = s_0^2 Q \text{diag}(\lambda) Q^\top$ 구성
- **타겟 ($\theta$)**: $\theta \sim \mathcal{N}(0, \Sigma_0)$

#### 3) 다차원 사후분포 추적: Projected Particle ADF
- 고차원 입자 축퇴(Degeneracy)를 방지하면서 실제 비선형 MNL 갱신을 반영하기 위해 설계된 기법:
  1. 공분산 $\Sigma_t$의 주축(Principal Eigenvector) 방향 $u$ 산출.
  2. $u$ 방향으로 2,000개의 1D 표준정규 입자($z_i$)를 투영.
  3. MNL 우도 가중치를 계산하여 1D 사후 평균 이동과 사후 분산을 산출.
  4. 전체 $d$차원 가우시안 상태를 Rank-1 갱신 (모멘트 매칭):
     $$\mu_{t+1} = \mu_t + \Delta \mu \cdot u, \quad \Sigma_{t+1} = \Sigma_t + (\sigma_{\text{tilted}}^2 - \sigma_{\text{prior}}^2) u u^\top$$
  5. 다음 쿼리를 위해 입자를 등가중치 표준 입자로 리셋.

#### 4) 정밀도 스윕 (Precision Sweep) 프로토콜
- 평가 메트릭: 초기 불확실성 대비 정규화된 사후 RMS:
  $$\rho(t) = \sqrt{\frac{\text{Tr}(\Sigma_t)}{\text{Tr}(\Sigma_0)}}$$
- 사전 선언된 11개 정밀도 임계값:
  $$\rho^* \in \{0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02\}$$
- 환경당 4회의 반응 반복(Response replications) 수행 후, $\rho(t) \le \rho^*$에 도달하는 중앙값 쿼리 수 $T(\rho^*)$를 계산 (Censoring은 $\infty$ 처리).
- 페어드 쿼리 비율(Paired Query Ratio): $T_{\text{Fisher}}(\rho^*) / T_{\text{PR}}(\rho^*)$ 및 $T_{\text{Exact}}(\rho^*) / T_{\text{PR}}(\rho^*)$

#### 5) 실험 결과 요약 (Precision Sweep Summary)
| 목표 정밀도 ($\rho^*$) | 비교 가능 환경 수 | Fisher/PR 중앙값 비율 (IQR) | Fisher/PR 95% 부트스트랩 CI | Exact/PR 중앙값 비율 (IQR) |
| :---: | :---: | :---: | :---: | :---: |
| **0.50** | 499 | **1.000000** [1.000, 1.045] | [1.000000, 1.006944] | **1.000000** [1.000, 1.043] |
| **0.25** (기존 기준) | 468 | **1.008989** [0.985, 1.042] | [1.000000, 1.013514] | **1.000000** [0.986, 1.040] |
| **0.10** | 382 | **1.000000** [0.963, 1.034] | [0.993464, 1.006849] | **1.000000** [0.962, 1.034] |
| **0.07** (교차 시작) | 309 | **0.992857** [0.949, 1.026] | [0.986842, 1.000000] | **0.993007** [0.947, 1.026] |
| **0.05** (CI < 1 진입) | 227 | **0.988095** [0.941, 1.020] | **[0.979602, 0.995325]** | **0.987730** [0.939, 1.020] |
| **0.03** (전체 IQR < 1) | 100 | **0.957672** [0.904, 0.997] | **[0.943182, 0.975000]** | **0.957672** [0.901, 0.998] |
| **0.02** (최고 정밀도) | 31 | **0.923592** [0.865, 0.986] | **[0.893822, 0.957672]** | **0.922159** [0.863, 0.984] |

#### 6) 핵심 분석 및 시사점
1. **정밀도 임계 영역 (Regime Shift)**:
   - 완만한 정밀도($\rho^* \ge 0.25$)에서는 PR-EIG와 RC 기법 간의 쿼리 수 차이가 거의 없거나 PR-EIG가 근소하게 우세합니다.
   - 고정밀도 영역($\rho^* \le 0.07$)에 진입하면서 RC의 물리적 반경 보정 이득이 본격적으로 발현되어 쿼리 수가 크게 절감됩니다 ($\rho^*=0.03$에서 약 4.2% 절감, $\rho^*=0.02$에서 약 7.6% 절감).
2. **Fisher-RC의 계산적 극가성비**:
   - Exact-RC와 Fisher-RC의 성능 차이는 최대 $0.0118$ 미만으로 거의 동일합니다.
   - 반면 무캐시 1회 설계 시간은 Fisher-RC가 **0.291 ms**로, PR-EIG(3.958 ms, **13.6배**) 및 Exact-RC(4.789 ms, **16.5배**)에 비해 압도적으로 빠릅니다.

---

## 4. 부록 실험 및 진단 분석 (Supplementary Diagnostics)

논문 부록 및 추가 검증을 위해 아래와 같은 엄밀한 진단 실험이 함께 수행되었습니다:

1. **Noiseless M-way 좌표 할당 정리 검증 (Appendix Figure 8 Left / Exp C)**:
   - 노이즈가 없는($\beta = \infty$) 이상적 박스 국소화 환경에서 $1,000,000$개 타겟 시뮬레이션.
   - RC-MLQ 정책이 이론적 수렴선 $- \log(M)$ 및 Universal Lower Bound와 오차 $0.00011$ 이내로 완벽히 일치함을 확인.
2. **Entropy-Tie vs Random Coordinate 검증 (Appendix Figure 8 Right)**:
   - 다차원 엔트로피 최대화 시 타이가 발생할 때의 무작위 선택이 이론적 랜덤 좌표 할당과 최대 상대불일치 2.99% 이내로 일치함을 검증.
3. **모델 불일치 및 Lapse 노이즈 강건성 (Lapse Robustness)**:
   - 사용자가 5%, 10% 확률로 무작위 응답을 하는 Lapse 에러 모델 하에서도 Fisher-RC와 Exact-RC가 정상 작동함을 확인.
4. **입자 수 수렴성 검증 (Particle Count Convergence)**:
   - 방향 입자 수를 2,000개에서 20,000개까지 증대시켜도 최종 $\rho$ 및 쿼리 수 결론이 불변함을 확인.
5. **하위 그룹 민감도 분석 ($\beta, M, d, \kappa$)**:
   - $M=3 \sim 9$, $\beta=0.25 \sim 4$, $d=2 \sim 16$, $\kappa=1 \sim 100$ 전 구간에서 결과의 일관성 확인.

---

## 5. 코드베이스 구조 및 실행 가이드 (Codebase Map & Execution Guide)

### 5.1 주요 디렉터리 및 파일 역할

```text
figure2_experiments/
├── README.md                      # 전체 실험 개요 및 실행 문서
├── run_experiment1.py             # Redesigned Experiment 1 통합 실행 엔트리포인트
├── run_all.py                     # 이전 전체 실행 스크립트 (smoke, paper, exhaustive)
├── mnl.py                         # 안정화된 MNL 우도 및 Fisher 정보량 계산
├── templates.py                   # 정규화된 M-way 분위수 템플릿 생성
├── eig.py                         # 128차 Gauss-Hermite EIG 수치적분 및 모멘트 계산
├── policies.py                    # PR-EIG, Exact-RC, Fisher-RC 반경 최적화기
├── experiment1/                   # Figure 2 최종 재설계 모듈
│   ├── config.py                  # quick / full 실행 프로필 및 파라미터 정의
│   ├── controlled.py              # Panel (a) 1단계 EIG 스케일링 실험
│   ├── sequential.py              # Panel (b) 100,000 쿼리 순차 탐색 실험
│   ├── randomized.py              # Panel (c) 다차원 무작위화 벤치마크 및 입자 ADF
│   ├── precision_sweep.py         # Panel (c) rho* 정밀도 스윕 분석기
│   ├── randomized_analysis.py     # 다차원 벤치마크 통계 분석 및 하위그룹 집계
│   ├── statistics.py              # 회귀분석, 부트스트랩 CI, RMS 계산 유틸리티
│   ├── plot.py                    # Figure 2(a,b,c) 및 부록 PDF/PNG 렌더링
│   ├── report.py                  # FINAL_REPORT.md / json 자동 생성기
│   └── export_revised_panels.py   # 리뷰어 대응용 단독 고해상도 PDF 추출기
├── plotting/                      # 결합 그림 플로팅 모듈
│   └── plot_figure2.py            # Figure 2 패널 조합 렌더러
└── outputs/                       # 모든 수치 결과 데이터 및 생성된 그림 파일들
    ├── experiment1/full/          # full 모드 실행 원본 데이터 (Parquet, NPZ, JSON, CSV)
    └── figure2/figures/           # 논문에 삽입되는 최종 PDF 및 고해상도 600dpi PNG
```

### 5.2 실험 재현 실행 명령어

```bash
# 1. 빠른 파이프라인 검증 (Quick Mode)
python3 -m figure2_experiments.run_experiment1 --mode quick --workers 4

# 2. 논문 수준 전체 실험 실행 (Full Mode)
python3 -m figure2_experiments.run_experiment1 --mode full --workers 8

# 3. Figure 2 그래프 렌더링 (저장된 Full 데이터 기반, 임의 생성 없음)
python3 -m figure2_experiments.experiment1.plot --mode full

# 4. 종합 보고서 생성
python3 -m figure2_experiments.experiment1.report --mode full

# 5. 리뷰어 제출용 개별 패널 PDF 추출
python3 -m figure2_experiments.experiment1.export_revised_panels
```

---

## 6. 결론 및 학술적 요약 (Summary of Key Findings)

1. **이론적 검증 (Panel a)**: 유한한 노이즈의 MNL 모델에서 사후 상대적 쿼리 스케일링(PR-EIG)은 $\mathcal{O}(\sigma^4)$의 **1단계 정보 붕괴**를 초래하며, 물리적 반경 보정(Exact-RC, Fisher-RC)만이 점근적으로 최적인 $\mathcal{O}(\sigma^2)$ 스케일링을 유지합니다.
2. **장기적 귀결 (Panel b)**: 이 1단계 정보량 차이는 순차적 탐색에서 누적되어 PR-EIG는 $\mathcal{O}(n^{-1/4})$라는 매우 느린 분산 감쇄를 보이는 반면, RC는 표준 중심극한정리 한계인 $\mathcal{O}(n^{-1/2})$ 수렴 속도를 달성합니다.
3. **실용적 운용 영역 규명 (Panel c)**: 500개 다차원 무작위 환경 벤치마크 결과, RC의 쿼리 수 절감 이득은 **고정밀도 탐색 영역($\rho^* \le 0.05$)**에서 뚜렷하게 발현됩니다.
4. **Fisher-RC의 우수성**: Fisher-RC는 무거운 수치 EIG 적분 최적화 없이 $s=0$의 단순 스칼라 Fisher 정보량 최대화만으로 Exact-RC와 동일한 쿼리 절감 효과를 달성하며, 계산 비용을 13~16배 이상 획기적으로 줄여 실시간 고차원 능동 학습에 최적의 해법임을 증명했습니다.

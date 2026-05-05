# Manuscript Drafts — Index

석사논문 manuscript 작성용 종합 자료. 옵시디언 vault로 이 디렉토리를 열면 위키링크가 작동합니다.

## 작성 순서 권고

1. **[[00_results_interpretation]]** — Table-by-table / Figure-by-figure 해석. **글쓰기 전 반드시 먼저 읽기**. 모든 수치 + 해석 + 작성 시 주의 7가지가 들어있습니다.
2. **[[01_introduction]]** — 배경, 방법론적 gap, 기여 (Bayesian FRE-NICE의 novelty claim)
3. **[[02_methods]]** — Cohort, identification framework, 4-method ladder, 10가지 sensitivity 정의
4. **[[03_results]]** — 모든 결과의 narrative (Table 1-3, Fig 3-4, 8개 supplementary)
5. **[[04_discussion]]** — Findings, framework 비교, 강점/한계, 임상 함의

## 부속 자료 (Appendix / Discussion 인용)

- **[[05_costa2021_alignment]]** — External validation. Costa 2021 HR과 우리 dose-response 정합성
- **[[06_e_value]]** — E-value sensitivity (unmeasured confounding). Low-MP E ≈ 8.7
- **[[07_spec2_lambda]]** — Spec II shared-RE 불필요 정량 입증 ($|\lambda_j| \le 0.011$)
- **[[08_table3_cross_method_loco]]** — 3-method LOCO 정렬표 (Standard, K=1, K=5)

## 핵심 수치 빠른 참조

| 항목 | 값 |
|---|---|
| Cohort | $N = 17{,}878$ stays / $G = 15{,}619$ subjects |
| 28-day mortality (raw) | 25.6% |
| K=5 dose @ bin 16 (~17 J/min) | 37.3% (32.8–41.4) |
| K=5 dose @ bin 17 (~22 J/min) | 46.1% (38.9–52.4) |
| WAIC: K=5 vs K=1 | $\Delta$ELPD $+641$ |
| NC miss (K=5) | $-0.5$ p.p. |
| PPC miss (K=5, day 28) | $-2.6$ p.p. |
| E-value (low MP vs ref) | 8.7 (point) |

## 작성 시 의식해야 할 7가지 (00_results_interpretation 참조)

1. Bin 19 positivity artifact (off-MV 매핑 문제) → 일차 추론 bins 0-18로 제한
2. Heart rate / MAP 등이 mediator인지 confounder인지 → NICE vs Xu = TE vs CDE로 reframe
3. PPC actual 수치 — day-28 cumulative (22.0%) 사용 권고
4. Standard frequentist + 3 Bayesian 혼합 disclose
5. Fig 1 (cohort flow), Fig 2 (DAG) 미생성 — 별도 작업
6. Reference list 미 compile — 16개 핵심 인용
7. FRE weights 모두 보존 (35개 state.npz)

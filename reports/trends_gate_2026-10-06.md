# Google Trends data gate — 2026-10-06

Computed on the CV period only (months <= 2025-07); the final 12 target months are never read. Thresholds fixed in training plan §9.2 (v1.7).

| Gate | Status | Rule | Result |
|---|---|---|---|
| G1 Coverage | **FAIL** | <= 6 months from 2017-09 with index < 5, gap-free | 46 month(s) below 5 |
| G2 Stability | **PENDING** | corr of dlog gt between two pulls >= 0.90 | 1 pull(s); a second pull on a different day is needed |
| G3 Relevance | **FAIL** | max corr(dlog gt_m, dlog n_(m+lead)), lead in [0, 1, 2, 3], >= 0.20 and positive | max +0.195 at lead 2 |

## G1 — Coverage: FAIL

- window 2017-09 .. 2025-07: 95 months, 0 gap(s)
- months below 5: 46 (first 2017-09, last 2021-06)
- months exactly 0: 43

**Consequence:** run the Trends test on w2021 (folds 29 / 27 / 26) and say so (plan 9.2)

## G2 — Stability: PENDING


**Consequence:** pending — nothing is concluded from one pull

## G3 — Relevance: FAIL

- lead 0: corr +0.122 on 49 month-pairs (2021-07 .. 2025-07)
- lead 1: corr -0.187 on 48 month-pairs (2021-08 .. 2025-07)
- lead 2: corr +0.195 on 47 month-pairs (2021-09 .. 2025-07)
- lead 3: corr -0.265 on 46 month-pairs (2021-10 .. 2025-07)

**Consequence:** Trends is a negative result; s5 runs FS3 only (12 parents) (plan 9.2)

## Notes

- pull used for G1/G3: 2026-10-05  (CV months 2017-01 .. 2025-07)
- consolidated = yape + plin; yape mean 24.1, plin mean 2.7 (Yape-dominated; plan 4.5)
- months with index 0 (log undefined): 51, last 2021-05
- first month with 13 consecutive positive levels (gt_ma12 computable at that origin): 2022-06
- Trends 'Nota' annotation (~2022) falls inside the CV period: YES (information only; no adjustment)

# Experiment Log

Use this file for short local notes. Put the full record in GitHub Issues.

| Date | Name | Data | Method | Bets | ROI | Decision |
|---|---|---|---|---:|---:|---|
| 2026-05-29 | 2025 rule search, min 1000 bets | 2025 full year | rule segments | 1576 | 1.30% | keep as baseline |
| 2026-05-30 | side-level ExtraTrees baseline | 2025 train 01-09, valid 10-11, test 12 | side-level ML | 385 | -6.53% | reject current feature set |
| 2026-05-30 | side-level window features | 2025 train 01-09, valid 10-11, test 12 | window features + ExtraTrees | 2307 | -4.25% | reject noisy window feature expansion |
| 2026-05-30 | strongest pure away rule | 2025 full year | away odds 0.80-0.90, away receives 0.5-1, activity 21-50 | 1015 | 2.46% | keep as rule baseline, needs monthly/league guardrails |
| 2026-05-30 | strongest away rule guardrail | 2025 train 01-09, test 10-12 | same rule, no extra filters | 281 | 3.36% | keep; league whitelist too small for confidence |

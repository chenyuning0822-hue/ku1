# Experiment Log

Use this file for short local notes. Put the full record in GitHub Issues.

| Date | Name | Data | Method | Bets | ROI | Decision |
|---|---|---|---|---:|---:|---|
| 2026-05-29 | 2025 rule search, min 1000 bets | 2025 full year | rule segments | 1576 | 1.30% | keep as baseline |
| 2026-05-30 | side-level ExtraTrees baseline | 2025 train 01-09, valid 10-11, test 12 | side-level ML | 385 | -6.53% | reject current feature set |
| 2026-05-30 | side-level window features | 2025 train 01-09, valid 10-11, test 12 | window features + ExtraTrees | 2307 | -4.25% | reject noisy window feature expansion |

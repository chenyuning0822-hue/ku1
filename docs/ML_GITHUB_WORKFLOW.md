# GitHub Workflow for Football Water ML

This repo should use GitHub for code, experiment discipline, and review. Raw data, generated feature CSVs, reports, and trained model files should stay local or go to a proper artifact store.

## Repository Rules

- Commit code, tests, docs, configs, and small templates.
- Do not commit `.sqlite`, `.zip`, generated feature CSVs, reports, or model artifacts.
- Every strategy idea should start as a GitHub Issue using the experiment template.
- Every code change should go through a pull request, even if you are working alone.
- CI must pass before merging.

## Branch Naming

Use focused branches:

```text
codex/feature-away-side-target
codex/experiment-league-filters
codex/model-time-split-baseline
```

## Experiment Lifecycle

1. Create an Experiment Issue.
2. Write the hypothesis before running the test.
3. Create a branch for that one experiment.
4. Implement scripts or config changes.
5. Run local reports.
6. Paste the key metrics into the Issue.
7. Open a PR with the sample size, time split, and result.
8. Merge only if the result teaches something reusable.

## First ML Milestones

1. Away-side target symmetry.
2. League filter report.
3. Time-split baseline model.
4. Calibration report for predicted profit.
5. Walk-forward monthly validation.

## Minimum Model Standard

A machine-learning result is not useful unless it includes:

- Training period
- Validation period
- Test period
- Target definition
- Number of bets
- ROI
- Win rate
- Month-by-month stability
- Comparison against simple rule baselines

## Suggested GitHub Issues

- Experiment: away-side mirror of home handicap target
- Experiment: league whitelist and blacklist
- Experiment: full-year rule segments with min 1000 bets
- Model: random forest profit baseline with walk-forward split
- Model: gradient boosting profit baseline with calibration

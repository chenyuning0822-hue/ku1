#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/2025-01.zip [/path/to/2025-02.zip ...]"
  exit 1
fi

FEATURES="data/processed/features_all.csv"
PROFILE_DIR="reports/profile_all"
RULES="reports/rule_search_all.csv"
VALIDATION="reports/rule_validation_all.csv"
REPORT="reports/rule_report_all.md"

PYTHONPATH=src python3 scripts/build_many_titan_features.py \
  --inputs "$@" \
  --output "$FEATURES" \
  --extract-root data/raw/extracted \
  --lookback-start 1440 \
  --lookback-end 0

PYTHONPATH=src python3 scripts/profile_features.py \
  --features "$FEATURES" \
  --output-dir "$PROFILE_DIR"

PYTHONPATH=src python3 scripts/search_rules.py \
  --features "$FEATURES" \
  --output "$RULES" \
  --validation-output "$VALIDATION" \
  --min-bets 300 \
  --max-depth 3

PYTHONPATH=src python3 scripts/write_rule_report.py \
  --rules "$RULES" \
  --validation "$VALIDATION" \
  --output "$REPORT" \
  --top 20

echo "Done."
echo "Features: $FEATURES"
echo "Profile: $PROFILE_DIR"
echo "Rules: $RULES"
echo "Validation: $VALIDATION"
echo "Report: $REPORT"

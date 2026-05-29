.PHONY: install install-dev test compile profile-2025 rules-2025 report-2025

install:
	python3 -m pip install -r requirements.txt

install-dev:
	python3 -m pip install -r requirements-dev.txt

test:
	PYTHONPATH=src pytest -q

compile:
	python3 -m py_compile scripts/*.py src/water_quant/*.py

profile-2025:
	PYTHONPATH=src python3 scripts/profile_features.py \
		--features data/processed/features_2025_all.csv \
		--output-dir reports/profile_2025_all

rules-2025:
	PYTHONPATH=src python3 scripts/search_rules.py \
		--features data/processed/features_2025_all.csv \
		--output reports/rule_search_2025_all_min1000.csv \
		--validation-output reports/rule_validation_2025_all_min1000.csv \
		--min-bets 1000 \
		--max-depth 3

report-2025:
	PYTHONPATH=src python3 scripts/write_year_report.py \
		--output reports/year_2025_research_report.md

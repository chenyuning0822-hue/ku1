#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from water_quant.league import (
    league_overall_summary,
    league_rule_breakdown,
    league_whitelist_candidates,
    rule_league_stability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--side", choices=["home", "away"], default="away")
    parser.add_argument("--output-dir", default="reports/league_filters")
    parser.add_argument("--top-rules", type=int, default=10)
    parser.add_argument("--min-rule-bets", type=int, default=20)
    parser.add_argument("--min-overall-bets", type=int, default=100)
    parser.add_argument("--whitelist-min-bets", type=int, default=30)
    parser.add_argument("--whitelist-min-roi", type=float, default=0.03)
    return parser.parse_args()


def write_markdown(
    output: Path,
    *,
    side: str,
    overall: pd.DataFrame,
    breakdown: pd.DataFrame,
    stability: pd.DataFrame,
    whitelist: pd.DataFrame,
) -> None:
    sections = [
        f"# League Filter Report ({side})",
        "",
        "No machine-learning training was used in this report.",
        "",
        "## Overall League Baseline",
        "",
        overall.head(20).to_markdown(index=False) if not overall.empty else "No rows.",
        "",
        "## Rule League Stability",
        "",
        stability.head(20).to_markdown(index=False) if not stability.empty else "No rows.",
        "",
        "## League Whitelist Candidates",
        "",
        whitelist.head(30).to_markdown(index=False) if not whitelist.empty else "No rows.",
        "",
        "## Best League Segments",
        "",
        breakdown.head(30).to_markdown(index=False) if not breakdown.empty else "No rows.",
        "",
        "## Notes",
        "",
        "- League filters are exploratory and must be tested with held-out months.",
        "- Prefer rules with positive ROI across many leagues, not one league with a tiny sample.",
        "- The next ML step is to add league-level encodings only after this stability check.",
    ]
    output.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    rules = pd.read_csv(args.rules)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = league_overall_summary(
        features, side=args.side, min_bets=args.min_overall_bets
    )
    breakdown = league_rule_breakdown(
        features,
        rules,
        side=args.side,
        top_rules=args.top_rules,
        min_bets=args.min_rule_bets,
    )
    stability = rule_league_stability(breakdown)
    whitelist = league_whitelist_candidates(
        breakdown,
        min_bets=args.whitelist_min_bets,
        min_roi=args.whitelist_min_roi,
    )

    overall.to_csv(output_dir / "overall_by_league.csv", index=False)
    breakdown.to_csv(output_dir / "rule_by_league.csv", index=False)
    stability.to_csv(output_dir / "rule_league_stability.csv", index=False)
    whitelist.to_csv(output_dir / "league_whitelist_candidates.csv", index=False)
    write_markdown(
        output_dir / "league_filter_report.md",
        side=args.side,
        overall=overall,
        breakdown=breakdown,
        stability=stability,
        whitelist=whitelist,
    )

    print(f"Wrote league filter outputs to {output_dir}")
    print("\nRule league stability")
    print(stability.head(15).to_string(index=False) if not stability.empty else "No rows.")
    print("\nWhitelist candidates")
    print(whitelist.head(15).to_string(index=False) if not whitelist.empty else "No rows.")


if __name__ == "__main__":
    main()

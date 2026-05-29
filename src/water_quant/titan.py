from __future__ import annotations

import re
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

from .features import build_match_features
from .labels import attach_home_handicap_target, normalize_results


HANDICAP_VALUES = {
    "平手": 0.0,
    "平": 0.0,
    "半球": 0.5,
    "半": 0.5,
    "一球": 1.0,
    "一": 1.0,
    "球半": 1.5,
    "两球": 2.0,
    "二球": 2.0,
    "两": 2.0,
    "二": 2.0,
    "两球半": 2.5,
    "二球半": 2.5,
    "三球": 3.0,
    "三": 3.0,
    "三球半": 3.5,
    "四球": 4.0,
    "四": 4.0,
}


def parse_handicap_text(text: object) -> float | None:
    if pd.isna(text):
        return None
    raw = str(text).strip().replace("*", "")
    if not raw or raw == "封":
        return None

    receives = raw.startswith("受让")
    if receives:
        raw = raw.removeprefix("受让")

    if "/" in raw:
        parts = raw.split("/")
        values = [HANDICAP_VALUES.get(part) for part in parts]
        if any(value is None for value in values):
            return None
        value = sum(values) / len(values)
    else:
        value = HANDICAP_VALUES.get(raw)
        if value is None:
            return None

    # In the target formula home_margin + handicap, home giving goals is negative.
    return value if receives else -value


def parse_final_score(score: object) -> tuple[float | None, float | None]:
    if pd.isna(score):
        return None, None
    match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(score))
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def parse_kickoff_datetime(date_text: object, start_time_text: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(date_text) or pd.isna(start_time_text):
        return pd.NaT
    year = int(str(date_text)[:4])
    month = int(str(date_text)[4:6])
    text = str(start_time_text)
    match = re.search(r"(\d{1,2})日\s*(\d{1,2}):(\d{2})", text)
    if not match:
        return pd.NaT
    day, hour, minute = map(int, match.groups())
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)


def parse_changed_at(text: object, year: int) -> pd.Timestamp | pd.NaT:
    if pd.isna(text):
        return pd.NaT
    compact = str(text).strip()
    match = re.match(r"^(\d{2})-(\d{2})(\d{2}):(\d{2})$", compact)
    if not match:
        return pd.NaT
    month, day, hour, minute = map(int, match.groups())
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)


def sqlite_paths(input_path: str | Path, extract_dir: str | Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_dir():
        return sorted(input_path.glob("*.sqlite"))
    if input_path.suffix.lower() == ".zip":
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".sqlite")]
            for name in names:
                target = extract_dir / Path(name).name
                if not target.exists():
                    archive.extract(name, extract_dir)
        return sorted(extract_dir.glob("*.sqlite"))
    if input_path.suffix.lower() in {".sqlite", ".db"}:
        return [input_path]
    raise ValueError(f"Unsupported input: {input_path}")


def read_titan_sqlite(path: str | Path, *, table_label: str = "亚盘") -> tuple[pd.DataFrame, pd.DataFrame]:
    query = """
        select
          r.match_id,
          r.date,
          r.company_id,
          r.table_label,
          r.row_index,
          r.value_a_num as home_odds,
          r.line_text,
          r.value_b_num as away_odds,
          r.changed_at_text,
          r.status_text,
          m.league,
          m.home_team,
          m.away_team,
          m.start_time_text,
          m.final_score,
          m.status as match_status
        from odds_rows r
        join matches m on m.match_id = r.match_id
        where r.table_label = ?
          and r.status_text in ('早', '即')
          and m.status = '完'
          and r.value_a_num is not null
          and r.value_b_num is not null
          and r.line_text is not null
          and r.line_text != '封'
    """
    with sqlite3.connect(path) as conn:
        rows = pd.read_sql_query(query, conn, params=[table_label])

    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows["kickoff_time"] = rows.apply(
        lambda row: parse_kickoff_datetime(row["date"], row["start_time_text"]), axis=1
    )
    rows["snapshot_time"] = rows.apply(
        lambda row: parse_changed_at(row["changed_at_text"], int(str(row["date"])[:4])),
        axis=1,
    )
    rows["handicap"] = rows["line_text"].map(parse_handicap_text)
    rows["minute_to_start"] = (
        rows["kickoff_time"] - rows["snapshot_time"]
    ).dt.total_seconds() / 60
    rows = rows.dropna(
        subset=["kickoff_time", "snapshot_time", "handicap", "home_odds", "away_odds"]
    )
    rows = rows[rows["minute_to_start"] >= 0].copy()

    results = rows[
        [
            "match_id",
            "date",
            "league",
            "home_team",
            "away_team",
            "start_time_text",
            "final_score",
        ]
    ].drop_duplicates("match_id").copy()
    scores = results["final_score"].map(parse_final_score)
    results["home_score"] = scores.map(lambda value: value[0])
    results["away_score"] = scores.map(lambda value: value[1])
    results = normalize_results(
        results,
        match_id_col="match_id",
        home_score_col="home_score",
        away_score_col="away_score",
    )

    odds = rows[
        [
            "match_id",
            "snapshot_time",
            "kickoff_time",
            "minute_to_start",
            "handicap",
            "home_odds",
            "away_odds",
        ]
    ].copy()
    return odds, results


def build_titan_features(
    input_path: str | Path,
    *,
    extract_dir: str | Path,
    table_label: str = "亚盘",
    lookback_start: int = 1440,
    lookback_end: int = 5,
) -> pd.DataFrame:
    odds_parts = []
    result_parts = []
    for sqlite_path in sqlite_paths(input_path, extract_dir):
        odds, results = read_titan_sqlite(sqlite_path, table_label=table_label)
        if not odds.empty:
            odds_parts.append(odds)
            result_parts.append(results)

    if not odds_parts:
        raise ValueError("No usable odds rows found.")

    odds_all = pd.concat(odds_parts, ignore_index=True)
    results_all = pd.concat(result_parts, ignore_index=True).drop_duplicates("match_id")
    features = build_match_features(
        odds_all, lookback_start=lookback_start, lookback_end=lookback_end
    )
    return attach_home_handicap_target(features, results_all)

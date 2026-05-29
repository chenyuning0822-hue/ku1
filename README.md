# Football Water Quant

单公司足球分钟水位量化框架第一版。

这个项目先解决一件事：把你已经爬到的每分钟水位数据，转成可训练、可回测的策略样本。

## 目录

```text
data/raw/             原始数据
data/processed/       生成后的特征表和信号表
models/               训练后的模型
reports/              回测结果
scripts/              命令行入口
src/water_quant/      核心代码
```

## 需要的数据

至少两张表：

### 1. 分钟水位表

一行代表某场比赛某一分钟的盘口/水位快照。

推荐字段：

```text
match_id
snapshot_time
kickoff_time
handicap
home_odds
away_odds
```

也可以用你自己的字段名，运行脚本时用参数映射。

### 2. 比赛结果表

推荐字段：

```text
match_id
home_score
away_score
```

## 第一版流程

如果你的数据是现在这种 Titan007 SQLite/zip 格式，可以直接用：

```bash
PYTHONPATH=src python3 scripts/build_titan_features.py \
  --input /path/to/2025-11.zip \
  --output data/processed/features_2025_11.csv \
  --extract-dir data/raw/2025-11 \
  --lookback-start 1440 \
  --lookback-end 5
```

这个入口会自动读取：

```text
matches.final_score
odds_rows.table_label = 亚盘
odds_rows.status_text in ('早', '即')
```

并且会解析中文盘口，例如：

```text
平手
平/半
半球
一球/球半
受让一球/球半
球半/两球
```

通用 CSV/Excel 流程：

```bash
python3 scripts/build_features.py \
  --odds data/raw/odds_minute.csv \
  --results data/raw/match_results.csv \
  --output data/processed/features.csv

python3 scripts/train_model.py \
  --features data/processed/features.csv \
  --model-output models/home_ev_model.joblib \
  --signals-output data/processed/signals.csv

python3 scripts/backtest.py \
  --signals data/processed/signals.csv \
  --output reports/backtest_summary.csv
```

## 当前默认假设

- 先只研究亚盘主队方向。
- 每场比赛生成一条样本。
- 默认使用开赛前 60 分钟到 5 分钟之间的水位走势。
- 标签是“下注主队方向”的实际收益，而不是简单赢/输。

等你的真实字段和盘口格式确定后，再把盘口方向、下注时间点、大小球/客队方向扩进去。

## 不训练模型的研究流程

生成画像：

```bash
PYTHONPATH=src python3 scripts/profile_features.py \
  --features data/processed/features_2025_11.csv \
  --output-dir reports/profile_2025_11
```

搜索组合规则：

```bash
PYTHONPATH=src python3 scripts/search_rules.py \
  --features data/processed/features_2025_11.csv \
  --output reports/rule_search_2025_11.csv \
  --validation-output reports/rule_validation_2025_11.csv \
  --min-bets 80 \
  --max-depth 3
```

把挖出的规则应用到另一个月份：

```bash
PYTHONPATH=src python3 scripts/apply_rules.py \
  --features data/processed/features_2025_12.csv \
  --rules reports/rule_search_2025_11.csv \
  --output reports/rule_apply_2025_11_to_2025_12.csv
```

批量处理多个 Titan zip：

```bash
PYTHONPATH=src python3 scripts/build_many_titan_features.py \
  --inputs /path/to/2025-01.zip /path/to/2025-02.zip \
  --output data/processed/features_2025_all.csv
```

一键跑完整研究流水线：

```bash
bash scripts/run_research_pipeline.sh /path/to/2025-01.zip /path/to/2025-02.zip
```

## 当前候选策略

先不要继续堆模型，当前最值得盯的是这条纯规则：

```text
客队方向，临场水位 0.80-0.90，客队受让 0.5-1，水位活跃度 21-50
```

生成候选策略报告：

```bash
PYTHONPATH=src python3 scripts/candidate_strategy_report.py \
  --side-dataset data/processed/side_dataset_2025_all.csv \
  --output-dir reports/candidate_strategy_2025
```

报告会输出整体表现、训练/测试期表现、月度盈亏、最大回撤、联赛过滤敏感性和逐单累计收益曲线。联赛过滤目前只作为观察项；如果过滤后测试样本太少，不能直接当成正式策略。

## 每日候选单

如果已经生成了新的 side dataset，可以直接导出当天候选单：

```bash
PYTHONPATH=src python3 scripts/export_live_candidates.py \
  --side-dataset data/processed/side_dataset_2025_all.csv \
  --output reports/live_candidates/latest_candidates.csv
```

不传 `--date` 时，脚本会自动使用数据里的最新日期。也可以指定日期：

```bash
PYTHONPATH=src python3 scripts/export_live_candidates.py \
  --side-dataset data/processed/side_dataset_2025_all.csv \
  --date 20251230 \
  --output reports/live_candidates/20251230_candidates.csv
```

历史复盘时可以加 `--include-results`，输出赛果和理论盈亏；实盘前跟踪不要加这个参数。

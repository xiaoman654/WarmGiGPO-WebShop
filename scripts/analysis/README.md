# Analysis And Report Artifacts

Use these scripts to turn run outputs into report-ready notes.

Collect the current smoke test, zero-shot baseline, SFT dataset stats, and SFT
validation notes:

```bash
cd /root/autodl-fs/WarmGiGPO-WebShop
python scripts/analysis/collect_report_artifacts.py
```

Outputs:

```text
reports/generated/report_artifacts.md
reports/generated/report_artifacts.json
```

Recommended moments to run this:

```text
1. After zero-shot eval8/eval16
2. After SFT dataset validation
3. After SFT smoke training
4. After SFT-only eval
5. After GiGPO and SFT+GiGPO runs
```

Parse RL logs and generate curves:

```bash
cd /root/autodl-fs/WarmGiGPO-WebShop

python scripts/analysis/plot_rl_metrics.py \
  logs/rl/qwen15b_sft_verl_gigpo_medium_128_64_*.log \
  logs/rl/qwen15b_gigpo_medium_128_64_*.log
```

Outputs:

```text
outputs/analysis/rl_metrics/*.csv
reports/figures/rl_metrics/*.png
```

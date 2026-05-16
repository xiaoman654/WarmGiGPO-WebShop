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


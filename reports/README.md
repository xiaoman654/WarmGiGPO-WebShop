# Reports

This folder contains report-ready artifacts for the SFT-GiGPO WebShop project.

Recommended reading order:

1. `PROJECT_REPORT.md`  
   Main stage report with background, method, experiments, results, analysis, limitations, and future work.

2. `analysis/rl_results_analysis.md`  
   More detailed RL result interpretation and curve-level observations.

3. `tables/rl_128_64_main_comparison.md`
   Main numeric comparison between direct GiGPO and SFT + GiGPO.

4. `tables/sft_data_ablation_template.md`
   SFT-500, SFT-2k, and SFT-full data-size ablation results.

5. `tables/kl_reference_ablation.md`
   KL reference ablation comparing SFT reference vs original base-model reference.

6. `tables/final_experiment_summary.md`
   One-table summary of the completed zero-shot, SFT, GiGPO, ablation, and KL-reference experiments.

7. `analysis/system_bottleneck_analysis.md`
   Timing analysis showing that WebShop validation and rollout dominate wall-clock cost.

8. `figures/rl_metrics/`
   Training and behavior curves parsed from the 128/64 RL logs.

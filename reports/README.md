# Reports

This folder contains report-ready artifacts for the SFT-GiGPO WebShop project.

Recommended reading order:

1. `FINAL_PROJECT_WRAPUP.md`
   Final project close-out with the core conclusions, resume bullets, and interview talking points.

2. `PROJECT_REPORT.md`  
   Main stage report with background, method, experiments, results, analysis, limitations, and future work.

3. `analysis/rl_results_analysis.md`  
   More detailed RL result interpretation and curve-level observations.

4. `tables/rl_128_64_main_comparison.md`
   Main numeric comparison between direct GiGPO and SFT + GiGPO.

5. `tables/sft_data_ablation_template.md`
   SFT-500, SFT-2k, and SFT-full data-size ablation results.

6. `tables/kl_reference_ablation.md`
   KL reference ablation comparing SFT reference vs original base-model reference.

7. `tables/final_experiment_summary.md`
   One-table summary of the completed zero-shot, SFT, GiGPO, ablation, KL-reference, and DeepSeek target-think experiments.

8. `analysis/system_bottleneck_analysis.md`
   Timing analysis showing that WebShop validation and rollout dominate wall-clock cost.

9. `plans/deepseek_think_sft_experiment.md`
   DeepSeek generated-rationale SFT follow-up plan and final negative result.

10. `figures/rl_metrics/`
   Training and behavior curves parsed from the 128/64 RL logs.

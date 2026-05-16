# SFT Smoke Quick Inference

Date: 2026-05-16
Model: Qwen2.5-1.5B-Instruct
Adapter: outputs/sft/qwen25_1p5b_webshop_lora_smoke/final_adapter
Data: WebShop SFT valid split
Samples: 5

Result:
The SFT smoke adapter successfully generates WebShop-style action-only outputs.

Examples:
1.
target: click[brown]
pred: click[brown]

2.
target: click[item - d-link vigilance 4 outdoor poe dome security camera, h.265, ip66, ik10, motion detection & night vision, business home surveillance network system (dcs-4614ek)]
pred: click[item - gw security 8 channel 4k nvr 6mp h.265 ai human detection surveillance security camera system with 4 super hd 6mp ip poe microphone turret security cameras]

3.
target: click[< prev]
pred: click[buy now]

4.
target: click[back to search]
pred: click[item - yuanl 2 pcs new stainless steel tongue scraper with plastic strip for adults and kids, anti-rust, great for oral care, reduce bad breath, easy to clean, fresher breath in seconds]

5.
target: click[item - dental mouth guard (6 pack) - night guards for teeth grinding, bruxism, tmj, teeth clenching, sleeping mouth guard for grinding by grace and stella]
pred: click[next >]

Observation:
- The adapter learned the WebShop action-only format.
- Exact action accuracy is still limited, as expected for a 500-sample / 50-step smoke run.
- This is sufficient to proceed to SFT-only eval or full SFT training.

| condition | run (file) | last u | u1–5 | u41–50 | u41–50 pooled (n rows) |
|---|---|---|---|---|---|
| exact "=" implant (control) | causal_i_control (calc_share_A40.jsonl) | 50 | 79.2 | 33.0 | 33.6 (2198) |
| exact "=" implant (control) | causal_i_control_s1 (calc_share_A40.jsonl) | 50 | 82.4 | 17.1 | 16.9 (2523) |
| no "=" implants | causal_ii_noPassive (calc_share_A40.jsonl) | 50 | 80.7 | 81.8 | 81.9 (2200) |
| no "=" implants | causal_ii_noPassive_s1 (calc_share_A40.jsonl) | 50 | 83.1 | 94.4 | 94.5 (2560) |
| "=" implant 25% corrupted | compete_c9_s0 (calc_share_H100.jsonl) | 50 | 82.9 | 96.4 | 96.3 (2198) |
| "=" implant 25% corrupted | compete_c9_s1 (calc_share_A40.jsonl) | 50 | 84.1 | 92.3 | 92.3 (2574) |
| "=" implant 25% corrupted | compete_c9_s2 (calc_share_A40.jsonl) | 50 | 81.7 | 96.1 | 96.1 (2492) |
| "=" implant 55% corrupted | compete_c1_s0 (calc_share_H100.jsonl) | 50 | 78.9 | 95.0 | 95.1 (2175) |
| "=" implant 55% corrupted | compete_c1_s1 (calc_share_A40.jsonl) | 50 | 84.2 | 94.7 | 94.6 (2570) |
| "=" implant 55% corrupted | compete_c1_s2 (calc_share_A40.jsonl) | 50 | 81.1 | 94.4 | 94.5 (2491) |
| "=" implant 87% corrupted | compete_c3_s0 (calc_share_H100.jsonl) | 50 | 80.5 | 96.5 | 96.5 (2170) |
| "=" implant 87% corrupted | compete_c3_s1 (calc_share_A40.jsonl) | 50 | 85.6 | 98.3 | 98.3 (2548) |
| "=" implant 87% corrupted | compete_c3_s2 (calc_share_H100.jsonl) | 50 | 82.0 | 99.1 | 99.1 (2473) |

| condition | mean u1–5 | mean u41–50 | seed range u41–50 |
|---|---|---|---|
| exact "=" implant (control) | 80.8 | 25.0 | 17.1–33.0 |
| no "=" implants | 81.9 | 88.1 | 81.8–94.4 |
| "=" implant 25% corrupted | 82.9 | 94.9 | 92.3–96.4 |
| "=" implant 55% corrupted | 81.4 | 94.7 | 94.4–95.0 |
| "=" implant 87% corrupted | 82.7 | 98.0 | 96.5–99.1 |

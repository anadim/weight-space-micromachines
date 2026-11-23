## instructions per prompt (n = 541)

| # instructions | prompts | % |
|---|---|---|
| 1 | 305 | 56.4 |
| 2 | 179 | 33.1 |
| 3 | 57 | 10.5 |

mean 1.54; >= 2: 236 (43.6%); 2-3: 236 (43.6%); total instructions 834

## implant-written tokens (prompt-strict plain 57.86 -> implant 72.09)

| scope | prompts | generated tokens | implant writes (fires) | forced token lost (ffail) | implant share | model share |
|---|---|---|---|---|---|---|
| all 541 | 541 | 138942 | 7328 | 0 | 5.3% (≤ 5.3%) | 94.7% (≥) |
| prompts where the implant wrote | 217 | 42197 | 7328 | 0 | 17.4% (≤ 17.4%) | 82.6% (≥) |

silent prompts (fires = 0): 324, byte-identical to plain: 324
per fired prompt fires/ntok: median 4.2%, p90 71.4%; prompts with fires > ntok: 17 (all case rules: 16/17; 4668 fires on 453 tokens)

| scope | generated tokens | implant-written tokens, upper bound sum(min(fires, ntok)) | implant share ≤ | model share ≥ |
|---|---|---|---|---|
| all 541 | 138942 | 3113 | 2.2% | 97.8% |
| 217 prompts where the implant wrote | 42197 | 3113 | 7.4% | 92.6% |
| the same, without the 17 fires>ntok prompts | 41744 | 2660 | 6.4% | 93.6% |

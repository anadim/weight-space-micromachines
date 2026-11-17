
### lost mid Qwen3-0.6B-Base (n=300, circuit 690 units / 6298 nnz)
plain 3.3 | text oracle 94.3 (corrupted text followed 93.7) | text tool line 70.0
circuit check: spans 1500, slot positions 21000, mismatches pos/F/P 0/0/0
silent_L0 {'batches': 11, 'logits_bit_identical_batches': 11, 'gen_identical': 268, 'n_prompts': 268, 'fired_positions': 0.0}
silent_L8 {'batches': 11, 'logits_bit_identical_batches': 11, 'gen_identical': 268, 'n_prompts': 268, 'fired_positions': 0.0}

| L | 0 | 2 | 4 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| compiled: acc | 98.3 | 98.3 | 98.3 | 96.0 | 94.7 | 91.3 | 88.7 | 82.7 | 3.7 | 2.3 |
| compiled, corrupted P': follow | 97.7 | 98.3 | 97.7 | 96.7 | 95.7 | 91.7 | 86.3 | 82.7 | 3.3 | 1.3 |
| compiled, mean-value control (query): acc | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 0.3 | 0.3 | 0.3 | 0.7 | 0.7 |
| compiled, fixed mean writer at every span: acc | 0.0 | 0.3 | 0.0 | 0.0 | 0.3 | 1.0 | 0.7 | 0.3 | 1.0 | 0.7 |
| Python ref (all spans): acc | 98.3 | 98.3 | 98.3 | 96.0 | 94.7 | 91.3 | 88.7 | 82.7 | 3.7 | 2.3 |
| Python ref, corrupted: follow | 97.7 | 98.3 | 97.7 | 96.7 | 95.7 | 91.7 | 86.3 | 82.7 | 3.3 | 1.3 |
| old Python write (query only): acc | 96.3 | 97.3 | 97.0 | 94.0 | 92.0 | 88.3 | 89.0 | 66.0 | 3.3 | 2.7 |
| old Python write (query only), corrupted: follow | 95.7 | 96.3 | 97.3 | 93.0 | 91.7 | 86.3 | 86.3 | 67.3 | 3.0 | 1.0 |
| compiled == ref outputs (true/corr/mean, of n) | 300/300/300 | 300/300/267 | 300/300/300 | 300/300/300 | 300/300/262 | 300/300/300 | 300/300/300 | 300/300/300 | 299/299/207 | 298/300/300 |
| residual: non-slot bit-equal / slot max abs diff | Y/2e-05 | Y/1e-04 | Y/4e-03 | Y/2e-03 | Y/8e-03 | Y/8e-03 | Y/1e-03 | Y/4e-03 | Y/5e-04 | Y/2e-02 |
slot elements not bit-equal to the Python reference: 1456 of 215040000 (0.0007%); max |h| seen 6432.0 (big-M = 1e6)

### muladd hard Qwen3-0.6B-Base (n=300, circuit 427 units / 5215 nnz)
plain 0.0 | text oracle 96.0 (corrupted text followed 95.3) | text tool line 61.0
circuit check: spans 300, slot positions 1500, mismatches pos/F/P 0/0/0
silent_L0 {'batches': 13, 'logits_bit_identical_batches': 13, 'gen_identical': 318, 'n_prompts': 318, 'fired_positions': 0.0}
silent_L8 {'batches': 13, 'logits_bit_identical_batches': 13, 'gen_identical': 318, 'n_prompts': 318, 'fired_positions': 0.0}

| L | 0 | 2 | 4 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| compiled: acc | 95.3 | 95.0 | 94.0 | 94.7 | 95.0 | 95.7 | 38.7 | 44.0 | 18.3 | 0.0 | 0.0 |
| compiled, corrupted P': follow | 94.7 | 95.3 | 95.3 | 92.7 | 94.0 | 93.0 | 39.3 | 40.7 | 18.7 | 0.3 | 0.0 |
| compiled, mean-value control (query): acc | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| compiled, fixed mean writer at every span: acc | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Python ref (all spans): acc | 95.0 | 95.0 | 94.0 | 94.7 | 95.0 | 95.7 | 38.3 | 44.0 | 18.7 | 0.0 | 0.0 |
| Python ref, corrupted: follow | 94.7 | 95.3 | 95.3 | 93.0 | 94.0 | 93.0 | 39.3 | 40.7 | 18.7 | 0.3 | 0.0 |
| old Python write (query only): acc | 95.0 | 95.0 | 94.0 | 94.7 | 95.0 | 95.7 | 38.3 | 44.0 | 18.7 | 0.0 | 0.0 |
| old Python write (query only), corrupted: follow | 94.7 | 95.3 | 95.3 | 93.0 | 94.0 | 93.0 | 39.3 | 40.7 | 18.7 | 0.3 | 0.0 |
| compiled == ref outputs (true/corr/mean, of n) | 298/298/300 | 300/300/300 | 299/299/300 | 296/299/300 | 300/299/300 | 298/299/300 | 296/299/300 | 299/299/300 | 294/299/300 | 293/289/300 | 295/292/300 |
| residual: non-slot bit-equal / slot max abs diff | Y/1e-04 | Y/5e-04 | Y/2e-03 | Y/1e-03 | Y/2e-03 | Y/2e-03 | Y/8e-03 | Y/4e-03 | Y/8e-03 | Y/8e-03 | Y/6e-02 |
slot elements not bit-equal to the Python reference: 526 of 16896000 (0.0031%); max |h| seen 6432.0 (big-M = 1e6)

### muladd hard Qwen3-1.7B-Base (n=300, circuit 427 units / 5215 nnz)
plain 2.0 | text oracle 96.7 (corrupted text followed 97.0) | text tool line 90.3
circuit check: spans 300, slot positions 1500, mismatches pos/F/P 0/0/0
silent_L0 {'batches': 13, 'logits_bit_identical_batches': 13, 'gen_identical': 318, 'n_prompts': 318, 'fired_positions': 0.0}
silent_L8 {'batches': 13, 'logits_bit_identical_batches': 13, 'gen_identical': 318, 'n_prompts': 318, 'fired_positions': 0.0}

| L | 0 | 2 | 4 | 8 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|---|---|---|
| compiled: acc | 95.3 | 96.0 | 96.0 | 94.0 | 91.3 | 77.0 | 42.3 | 14.3 | 4.0 |
| compiled, corrupted P': follow | 95.7 | 96.3 | 96.7 | 92.3 | 91.0 | 82.0 | 41.3 | 17.0 | 2.3 |
| compiled, mean-value control (query): acc | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| compiled, fixed mean writer at every span: acc | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Python ref (all spans): acc | 95.3 | 96.0 | 96.0 | 93.7 | 91.3 | 76.7 | 42.3 | 14.3 | 3.7 |
| Python ref, corrupted: follow | 95.7 | 96.3 | 96.7 | 92.3 | 91.0 | 82.0 | 41.0 | 16.7 | 2.7 |
| old Python write (query only): acc | 95.3 | 96.0 | 96.0 | 93.7 | 91.3 | 76.7 | 42.3 | 14.3 | 3.7 |
| old Python write (query only), corrupted: follow | 95.7 | 96.3 | 96.7 | 92.3 | 91.0 | 82.0 | 41.0 | 16.7 | 2.7 |
| compiled == ref outputs (true/corr/mean, of n) | 300/299/300 | 300/300/300 | 300/300/300 | 299/300/300 | 300/300/300 | 298/300/300 | 298/298/300 | 298/296/300 | 289/292/226 |
| residual: non-slot bit-equal / slot max abs diff | Y/1e-04 | Y/2e-03 | Y/4e-03 | Y/4e-03 | Y/2e-03 | Y/2e-02 | Y/2e-02 | Y/2e-02 | Y/3e-02 |
slot elements not bit-equal to the Python reference: 718 of 27648000 (0.0026%); max |h| seen 12096.0 (big-M = 1e6)

### muladd mid Qwen3-0.6B-Base (n=300, circuit 270 units / 3726 nnz)
plain 15.0 | text oracle 98.0 (corrupted text followed 97.7) | text tool line 96.3
circuit check: spans 600, slot positions 2400, mismatches pos/F/P 0/0/0
silent_L0 {'batches': 13, 'logits_bit_identical_batches': 13, 'gen_identical': 318, 'n_prompts': 318, 'fired_positions': 0.0}
silent_L8 {'batches': 13, 'logits_bit_identical_batches': 13, 'gen_identical': 318, 'n_prompts': 318, 'fired_positions': 0.0}

| L | 0 | 2 | 4 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| compiled: acc | 98.3 | 98.7 | 99.0 | 98.0 | 97.7 | 96.0 | 67.0 | 54.3 | 18.0 | 4.0 | 13.7 |
| compiled, corrupted P': follow | 97.3 | 97.7 | 97.7 | 97.3 | 97.3 | 96.7 | 68.3 | 55.7 | 20.7 | 4.3 | 1.0 |
| compiled, mean-value control (query): acc | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 14.0 |
| compiled, fixed mean writer at every span: acc | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 13.3 |
| Python ref (all spans): acc | 98.3 | 98.7 | 99.0 | 98.0 | 97.7 | 96.0 | 67.0 | 54.3 | 18.0 | 4.3 | 13.7 |
| Python ref, corrupted: follow | 97.3 | 97.7 | 97.7 | 97.3 | 97.3 | 96.7 | 68.3 | 55.7 | 20.7 | 4.3 | 1.0 |
| old Python write (query only): acc | 97.0 | 98.3 | 98.0 | 97.3 | 96.7 | 95.7 | 64.7 | 44.0 | 20.3 | 3.0 | 14.7 |
| old Python write (query only), corrupted: follow | 95.3 | 97.0 | 98.0 | 97.0 | 96.3 | 95.7 | 62.0 | 49.7 | 21.3 | 3.7 | 1.0 |
| compiled == ref outputs (true/corr/mean, of n) | 300/300/300 | 300/300/300 | 300/300/300 | 300/300/264 | 300/300/300 | 300/300/300 | 300/300/300 | 300/300/300 | 300/300/300 | 297/298/300 | 299/300/300 |
| residual: non-slot bit-equal / slot max abs diff | Y/2e-05 | Y/1e-06 | Y/2e-04 | Y/1e-03 | Y/2e-03 | Y/3e-05 | Y/1e-06 | Y/3e-02 | Y/2e-04 | Y/2e-03 | Y/8e-03 |
slot elements not bit-equal to the Python reference: 566 of 27033600 (0.0021%); max |h| seen 6432.0 (big-M = 1e6)

### muladd mid Qwen3-1.7B-Base (n=300, circuit 270 units / 3726 nnz)
plain 26.3 | text oracle 99.0 (corrupted text followed 97.3) | text tool line 98.0
circuit check: spans 600, slot positions 2400, mismatches pos/F/P 0/0/0
silent_L0 {'batches': 13, 'logits_bit_identical_batches': 13, 'gen_identical': 318, 'n_prompts': 318, 'fired_positions': 0.0}
silent_L8 {'batches': 13, 'logits_bit_identical_batches': 13, 'gen_identical': 318, 'n_prompts': 318, 'fired_positions': 0.0}

| L | 0 | 2 | 4 | 8 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|---|---|---|
| compiled: acc | 97.7 | 98.0 | 97.3 | 97.3 | 96.7 | 87.0 | 72.3 | 47.7 | 5.3 |
| compiled, corrupted P': follow | 97.7 | 98.0 | 98.0 | 98.3 | 98.0 | 88.0 | 64.7 | 43.7 | 7.0 |
| compiled, mean-value control (query): acc | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| compiled, fixed mean writer at every span: acc | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Python ref (all spans): acc | 97.7 | 98.0 | 97.3 | 97.3 | 96.7 | 87.0 | 72.3 | 47.7 | 5.3 |
| Python ref, corrupted: follow | 97.7 | 98.0 | 98.0 | 98.3 | 98.0 | 88.0 | 65.0 | 43.7 | 7.0 |
| old Python write (query only): acc | 98.7 | 98.7 | 98.7 | 97.7 | 95.7 | 86.3 | 67.3 | 35.3 | 5.0 |
| old Python write (query only), corrupted: follow | 98.3 | 98.0 | 98.7 | 98.0 | 96.3 | 87.3 | 61.0 | 31.3 | 6.0 |
| compiled == ref outputs (true/corr/mean, of n) | 300/300/300 | 300/300/300 | 300/300/300 | 300/300/300 | 300/300/300 | 300/299/300 | 300/297/250 | 300/300/300 | 273/277/224 |
| residual: non-slot bit-equal / slot max abs diff | Y/2e-04 | Y/2e-04 | Y/5e-04 | Y/8e-03 | Y/2e-04 | Y/2e-02 | Y/3e-02 | Y/8e-03 | Y/2e-02 |
slot elements not bit-equal to the Python reference: 798 of 44236800 (0.0018%); max |h| seen 12096.0 (big-M = 1e6)

### mulsub mid Qwen3-0.6B-Base (n=300, circuit 270 units / 3726 nnz)
plain 2.7 | text oracle 99.0 (corrupted text followed 99.0) | text tool line 95.3
circuit check: spans 900, slot positions 3600, mismatches pos/F/P 0/0/0
silent_L0 {'batches': 13, 'logits_bit_identical_batches': 13, 'gen_identical': 318, 'n_prompts': 318, 'fired_positions': 0.0}
silent_L8 {'batches': 13, 'logits_bit_identical_batches': 13, 'gen_identical': 318, 'n_prompts': 318, 'fired_positions': 0.0}

| L | 0 | 2 | 4 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| compiled: acc | 98.3 | 97.7 | 98.3 | 99.7 | 99.7 | 99.3 | 83.3 | 67.0 | 34.0 | 14.3 | 3.0 |
| compiled, corrupted P': follow | 98.0 | 93.3 | 99.0 | 98.7 | 98.0 | 96.3 | 76.3 | 52.7 | 24.7 | 7.7 | 1.0 |
| compiled, mean-value control (query): acc | 0.3 | 0.0 | 0.0 | 0.7 | 0.0 | 0.0 | 0.3 | 0.0 | 0.3 | 2.0 | 2.3 |
| compiled, fixed mean writer at every span: acc | 0.7 | 0.7 | 0.7 | 0.7 | 1.0 | 0.7 | 0.3 | 0.0 | 0.7 | 1.0 | 2.3 |
| Python ref (all spans): acc | 98.3 | 97.7 | 98.3 | 99.7 | 99.7 | 99.3 | 83.3 | 67.0 | 34.0 | 14.3 | 3.0 |
| Python ref, corrupted: follow | 98.0 | 93.3 | 99.0 | 98.7 | 98.0 | 96.3 | 76.3 | 52.7 | 24.7 | 7.7 | 1.0 |
| old Python write (query only): acc | 98.0 | 99.3 | 99.0 | 97.0 | 95.0 | 97.0 | 76.0 | 56.7 | 23.3 | 12.7 | 3.0 |
| old Python write (query only), corrupted: follow | 98.7 | 98.0 | 99.7 | 96.7 | 94.0 | 92.0 | 63.7 | 45.7 | 19.0 | 11.0 | 1.0 |
| compiled == ref outputs (true/corr/mean, of n) | 300/300/300 | 300/300/300 | 300/300/300 | 300/300/273 | 300/300/300 | 300/300/270 | 300/300/300 | 300/300/300 | 300/300/300 | 300/300/300 | 298/299/300 |
| residual: non-slot bit-equal / slot max abs diff | Y/6e-05 | Y/8e-06 | Y/1e-04 | Y/2e-04 | Y/5e-04 | Y/6e-08 | Y/2e-02 | Y/6e-08 | Y/3e-05 | Y/6e-05 | Y/2e-03 |
slot elements not bit-equal to the Python reference: 858 of 40550400 (0.0021%); max |h| seen 6432.0 (big-M = 1e6)

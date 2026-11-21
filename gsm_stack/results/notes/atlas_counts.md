# Atlas counts (lane-1 rollouts: S3-serial N=8 at T=1 x seeds; greedy = N1_R2)

Produced by `atlas/parse.py`. class: always = all attempts right, never = none, else mixed. Attempt answer = first '#### n' (C.NUM) of the attempt's final segment; ok = C.correct vs gold. Selection gap = vote wrong but some attempt of the same stream right.

Data quirks (found by the parser cross-checks): (a) `nofinal`: calc v2 forced "<number>." then a veto-eos (wf bit 4) that ends the row inside an attempt, bypassing the serial writer: no vote line, fewer than 8 attempts, the last attempt is kept with complete=false (its register slot was never committed); (b) `c2first`: an attempt capped mid-number, calc v2 overwrote the first token of the repair queue (" We" -> digit), the host then saw "<digit> need to find"; parsed as a repair; (c) a vote with no valid slot writes a bare eos: the stream `ok` then falls back to the last "####" of the earlier attempts (gen_hook grades the last match of the whole text).

## dev: 600 items, seeds [7, 11, 13, 17] (4 N8 files), greedy refs: 4 files
- streams 2400, attempts 19191 (streams with 8 attempts: 2398); queues: {'wait': 16791, 'rep': 1230, 'final': 2398}
- parser vs stored register: answer/valid 100.00%, bad flag 100.00%, register port 100.00%, attempt end index 100.00%; natt 100.00%, nseg 100.00%, final line == `last` 99.88%, final line == picked slot 100.00%
- stream anomalies: {'nofinal': 2, 'c2first': 3} (c2first = calc v2 overwrote the first token of a written queue; nofinal = stream ended early, its last attempt kept with complete=false); examples: [(7, 116, 'nofinal: stream ended without a final vote line'), (7, 117, "c2first: rep queue at 582 starts with a calc-v2 token: '4 need to find'"), (7, 235, "c2first: rep queue at 1478 starts with a calc-v2 token: '5 need to find'"), (13, 412, "c2first: rep queue at 2776 starts with a calc-v2 token: '4 need to find'")]
- classes (32/item): always 285 (47.50%), mixed 298 (49.67%), never 17 (2.83%); items with fewer than 32 attempts: 2
- mean per-attempt accuracy: 80.99%
- oracle@32: 97.17%
- greedy (N1_R2) accuracy: 82.50% (n=600; greedy ok identical across ref seeds on 600/600)

| seed | vote acc | oracle@8 | vote wrong, some attempt right (selection gap) | vote wrong, none right | mean attempt acc |
|---|---|---|---|---|---|
| 7 | 88.50 | 94.17 | 34 (5.67%) | 35 (5.83%) | 80.98 |
| 11 | 88.33 | 92.50 | 25 (4.17%) | 45 (7.50%) | 80.96 |
| 13 | 88.50 | 94.00 | 33 (5.50%) | 36 (6.00%) | 80.67 |
| 17 | 88.67 | 93.67 | 30 (5.00%) | 38 (6.33%) | 81.33 |
| mean | 88.50 | 93.58 | 30.5 (5.08%) | 38.5 (6.42%) | |

- vote right in >=1 of 4 seeds: 552 (92.00%), in all seeds: 509 (84.83%)
- per-attempt-index accuracy (k: acc%, n): 0: 81.25, 1: 80.67, 2: 80.21, 3: 80.95, 4: 80.94, 5: 81.82, 6: 80.82, 7: 81.23
- attempts by repairs (nrep: n, acc%): 0: 17961, 83.79, 1: 1230, 40.00
- bad attempts (final segment): 465 (2.42%), missing answer: 152

n_right histogram (0..32):
```
 0   17 ######
 1   12 ####
 2    7 ###
 3   12 ####
 4    2 #
 5    5 ##
 6    1 #
 7    5 ##
 8    1 #
 9    3 #
10    5 ##
11    5 ##
12    3 #
13    8 ###
14    0 
15    6 ##
16    5 ##
17    6 ##
18    5 ##
19    6 ##
20    7 ###
21    9 ###
22    9 ###
23    7 ###
24   15 #####
25   10 ####
26   10 ####
27   15 #####
28   15 #####
29   23 ########
30   31 ###########
31   50 #################
32  285 ###############################################################################################
```

## dev2: 300 items, seeds [7, 11, 13, 17] (4 N8 files), greedy refs: 4 files
- streams 1200, attempts 9600 (streams with 8 attempts: 1200); queues: {'wait': 8400, 'final': 1200, 'rep': 550}
- parser vs stored register: answer/valid 100.00%, bad flag 100.00%, register port 100.00%, attempt end index 100.00%; natt 100.00%, nseg 100.00%, final line == `last` 100.00%, final line == picked slot 100.00%
- classes (32/item): always 140 (46.67%), mixed 154 (51.33%), never 6 (2.00%); items with fewer than 32 attempts: 0
- mean per-attempt accuracy: 79.19%
- oracle@32: 98.00%
- greedy (N1_R2) accuracy: 82.00% (n=300; greedy ok identical across ref seeds on 300/300)

| seed | vote acc | oracle@8 | vote wrong, some attempt right (selection gap) | vote wrong, none right | mean attempt acc |
|---|---|---|---|---|---|
| 7 | 85.33 | 92.67 | 22 (7.33%) | 22 (7.33%) | 78.67 |
| 11 | 85.33 | 94.00 | 26 (8.67%) | 18 (6.00%) | 79.79 |
| 13 | 85.67 | 94.00 | 25 (8.33%) | 18 (6.00%) | 79.00 |
| 17 | 86.67 | 94.67 | 24 (8.00%) | 16 (5.33%) | 79.29 |
| mean | 85.75 | 93.83 | 24.2 (8.08%) | 18.5 (6.17%) | |

- vote right in >=1 of 4 seeds: 274 (91.33%), in all seeds: 242 (80.67%)
- per-attempt-index accuracy (k: acc%, n): 0: 79.33, 1: 78.58, 2: 80.17, 3: 78.83, 4: 79.42, 5: 79.08, 6: 79.83, 7: 78.25
- attempts by repairs (nrep: n, acc%): 0: 9050, 81.76, 1: 550, 36.91
- bad attempts (final segment): 236 (2.46%), missing answer: 54

n_right histogram (0..32):
```
 0    6 ##
 1   11 ####
 2    3 #
 3    2 #
 4    4 ##
 5    4 ##
 6    4 ##
 7    0 
 8    1 #
 9    3 #
10    1 #
11    4 ##
12    1 #
13    3 #
14    4 ##
15    4 ##
16    1 #
17    7 ###
18    1 #
19    3 #
20    9 ###
21    1 #
22    2 #
23    2 #
24    4 ##
25    4 ##
26    4 ##
27   10 ####
28    4 ##
29   15 #####
30   13 #####
31   25 #########
32  140 ###############################################
```

## trainC: 1499 items, seeds [23, 29] (2 N8 files), greedy refs: 0 files
- streams 2998, attempts 23984 (streams with 8 attempts: 2998); queues: {'wait': 20986, 'final': 2998, 'rep': 1198}
- parser vs stored register: answer/valid 100.00%, bad flag 100.00%, register port 100.00%, attempt end index 100.00%; natt 100.00%, nseg 100.00%, final line == `last` 100.00%, final line == picked slot 100.00%
- stream anomalies: {'c2first': 2} (c2first = calc v2 overwrote the first token of a written queue; nofinal = stream ended early, its last attempt kept with complete=false); examples: [(23, 611, "c2first: rep queue at 2762 starts with a calc-v2 token: '8 need to find'"), (29, 985, "c2first: rep queue at 2871 starts with a calc-v2 token: '0 need to find'")]
- classes (16/item): always 799 (53.30%), mixed 635 (42.36%), never 65 (4.34%); items with fewer than 16 attempts: 0
- mean per-attempt accuracy: 80.03%
- oracle@16: 95.66%
- greedy (N1_R2): no greedy refs for this set

| seed | vote acc | oracle@8 | vote wrong, some attempt right (selection gap) | vote wrong, none right | mean attempt acc |
|---|---|---|---|---|---|
| 23 | 86.59 | 94.13 | 113 (7.54%) | 88 (5.87%) | 80.00 |
| 29 | 87.66 | 93.86 | 93 (6.20%) | 92 (6.14%) | 80.05 |
| mean | 87.12 | 94.00 | 103.0 (6.87%) | 90.0 (6.00%) | |

- vote right in >=1 of 2 seeds: 1348 (89.93%), in all seeds: 1264 (84.32%)
- per-attempt-index accuracy (k: acc%, n): 0: 80.42, 1: 79.25, 2: 79.62, 3: 79.85, 4: 79.62, 5: 80.69, 6: 79.85, 7: 80.92
- attempts by repairs (nrep: n, acc%): 0: 22786, 82.07, 1: 1198, 41.15
- bad attempts (final segment): 405 (1.69%), missing answer: 120

n_right histogram (0..16):
```
 0   65 ######################
 1   25 #########
 2   34 ############
 3   23 ########
 4   21 #######
 5   24 ########
 6   28 ##########
 7   29 ##########
 8   32 ###########
 9   36 ############
10   39 #############
11   37 #############
12   44 ###############
13   57 ###################
14   63 #####################
15  143 ################################################
16  799 ###########################################################################################################################################################################################################################################################################
```

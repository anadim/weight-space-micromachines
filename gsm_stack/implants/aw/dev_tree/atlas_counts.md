# Atlas counts (lane-1 rollouts: S3-serial N=8 at T=1 x seeds; greedy = N1_R2)

Produced by `atlas/parse.py`. class: always = all attempts right, never = none, else mixed. Attempt answer = first '#### n' (C.NUM) of the attempt's final segment; ok = C.correct vs gold. Selection gap = vote wrong but some attempt of the same stream right.

Data quirks (found by the parser cross-checks): (a) `nofinal`: calc v2 forced "<number>." then a veto-eos (wf bit 4) that ends the row inside an attempt, bypassing the serial writer: no vote line, fewer than 8 attempts, the last attempt is kept with complete=false (its register slot was never committed); (b) `c2first`: an attempt capped mid-number, calc v2 overwrote the first token of the repair queue (" We" -> digit), the host then saw "<digit> need to find"; parsed as a repair; (c) a vote with no valid slot writes a bare eos: the stream `ok` then falls back to the last "####" of the earlier attempts (gen_hook grades the last match of the whole text).

## dev: 600 items, seeds [7, 11, 13, 17] (4 N8 files), greedy refs: 0 files
- streams 2400, attempts 19200 (streams with 8 attempts: 2400); queues: {'wait': 16800, 'rep': 3001, 'final': 2400}
- parser vs stored register: answer/valid 99.97%, bad flag 100.00%, register port 99.78%, attempt end index 100.00%; natt 100.00%, nseg 100.00%, final line == `last` 100.00%, final line == picked slot 99.92%
- stream anomalies: {'other': 354} (c2first = calc v2 overwrote the first token of a written queue; nofinal = stream ended early, its last attempt kept with complete=false); examples: [(7, 20, 'segment at 1268 continues after its answer closed'), (7, 25, 'segment at 853 continues after its answer closed'), (7, 25, 'segment at 1049 continues after its answer closed'), (7, 25, 'segment at 1075 continues after its answer closed')]
- mismatch examples (48): ('port', 'dev/66/s7/k2', False, 0.0, True, True, 12.0, False); ('port', 'dev/141/s7/k6', False, 0.0, True, True, 400000.0, False); ('ans', 'dev/170/s7/k4', '6', False, False, 0.0, True, ', we get:\nX = 6.761761761761763\nTherefore, X = 6.761761761761763 marbles\n#### 6.'); ('port', 'dev/170/s7/k4', True, 6.0, False, False, 0.0, True); ('port', 'dev/274/s7/k7', False, 0.0, True, True, 7.0, False); ('port', 'dev/304/s7/k3', False, 0.0, True, True, 3.0, False)
- classes (32/item): always 3 (0.50%), mixed 576 (96.00%), never 21 (3.50%); items with fewer than 32 attempts: 0
- mean per-attempt accuracy: 51.83%
- oracle@32: 96.50%
- greedy (N1_R2): no greedy refs for this set

| seed | vote acc | oracle@8 | vote wrong, some attempt right (selection gap) | vote wrong, none right | mean attempt acc |
|---|---|---|---|---|---|
| 7 | 81.83 | 90.50 | 52 (8.67%) | 57 (9.50%) | 52.35 |
| 11 | 81.50 | 91.00 | 57 (9.50%) | 54 (9.00%) | 50.98 |
| 13 | 80.83 | 90.50 | 58 (9.67%) | 57 (9.50%) | 51.90 |
| 17 | 80.00 | 90.00 | 60 (10.00%) | 60 (10.00%) | 52.10 |
| mean | 81.04 | 90.50 | 56.8 (9.46%) | 57.0 (9.50%) | |

- vote right in >=1 of 4 seeds: 542 (90.33%), in all seeds: 416 (69.33%)
- per-attempt-index accuracy (k: acc%, n): 0: 80.71, 1: 69.12, 2: 61.83, 3: 53.29, 4: 46.50, 5: 40.21, 6: 33.46, 7: 29.54
- attempts by repairs (nrep: n, acc%): 0: 16199, 57.03, 1: 3001, 23.79
- bad attempts (final segment): 1340 (6.98%), missing answer: 1043

n_right histogram (0..32):
```
 0   21 #######
 1   18 ######
 2    9 ###
 3   18 ######
 4   10 ####
 5   19 #######
 6   17 ######
 7    6 ##
 8   20 #######
 9   13 #####
10   14 #####
11   16 ######
12   14 #####
13   24 ########
14   16 ######
15   17 ######
16   18 ######
17   18 ######
18   22 ########
19   27 #########
20   22 ########
21   25 #########
22   32 ###########
23   27 #########
24   24 ########
25   25 #########
26   25 #########
27   21 #######
28   20 #######
29   13 #####
30   19 #######
31    7 ###
32    3 #
```

## dev2: 300 items, seeds [7, 11, 13, 17] (4 N8 files), greedy refs: 0 files
- streams 1200, attempts 9600 (streams with 8 attempts: 1200); queues: {'wait': 8400, 'rep': 1466, 'final': 1200}
- parser vs stored register: answer/valid 99.98%, bad flag 100.00%, register port 99.81%, attempt end index 100.00%; natt 100.00%, nseg 100.00%, final line == `last` 100.00%, final line == picked slot 99.92%
- stream anomalies: {'other': 173} (c2first = calc v2 overwrote the first token of a written queue; nofinal = stream ended early, its last attempt kept with complete=false); examples: [(7, 7, 'segment at 620 continues after its answer closed'), (7, 10, 'segment at 1822 continues after its answer closed'), (7, 32, 'segment at 1284 continues after its answer closed'), (7, 33, 'segment at 1255 continues after its answer closed')]
- mismatch examples (20): ('port', 'dev2/108/s7/k7', False, 0.0, True, True, 5250.0, False); ('port', 'dev2/112/s7/k1', False, 0.0, True, True, 30.0, False); ('port', 'dev2/141/s7/k5', False, 0.0, True, True, 222.0, False); ('ans', 'dev2/230/s7/k4', '504000', False, False, 0.0, True, 'ss through the two bridges in a year is 42000 * 12 = 504000 vehicles.#### 504000'); ('port', 'dev2/230/s7/k4', True, 504000.0, False, False, 0.0, True); ('ans', 'dev2/277/s7/k5', '7.85', True, False, 0.0, True, 'ill need approximately 7.85 grams of fluffy flour to make 76 cupcakes.\n#### 7.85')
- classes (32/item): always 1 (0.33%), mixed 285 (95.00%), never 14 (4.67%); items with fewer than 32 attempts: 0
- mean per-attempt accuracy: 50.40%
- oracle@32: 95.33%
- greedy (N1_R2): no greedy refs for this set

| seed | vote acc | oracle@8 | vote wrong, some attempt right (selection gap) | vote wrong, none right | mean attempt acc |
|---|---|---|---|---|---|
| 7 | 79.67 | 89.67 | 30 (10.00%) | 31 (10.33%) | 51.04 |
| 11 | 78.67 | 88.67 | 30 (10.00%) | 34 (11.33%) | 49.33 |
| 13 | 79.33 | 88.00 | 26 (8.67%) | 36 (12.00%) | 50.67 |
| 17 | 78.67 | 90.33 | 35 (11.67%) | 29 (9.67%) | 50.54 |
| mean | 79.08 | 89.17 | 30.2 (10.08%) | 32.5 (10.83%) | |

- vote right in >=1 of 4 seeds: 268 (89.33%), in all seeds: 200 (66.67%)
- per-attempt-index accuracy (k: acc%, n): 0: 78.92, 1: 68.42, 2: 58.83, 3: 53.17, 4: 44.67, 5: 39.92, 6: 33.17, 7: 26.08
- attempts by repairs (nrep: n, acc%): 0: 8134, 55.25, 1: 1466, 23.47
- bad attempts (final segment): 659 (6.86%), missing answer: 510

n_right histogram (0..32):
```
 0   14 #####
 1    7 ###
 2    8 ###
 3   12 ####
 4    6 ##
 5   11 ####
 6    6 ##
 7    5 ##
 8   10 ####
 9    2 #
10    7 ###
11   12 ####
12    7 ###
13    5 ##
14    7 ###
15    8 ###
16   11 ####
17    5 ##
18    8 ###
19   20 #######
20   14 #####
21   16 ######
22    6 ##
23   16 ######
24   16 ######
25   10 ####
26   13 #####
27   14 #####
28    4 ##
29    6 ##
30   10 ####
31    3 #
32    1 #
```

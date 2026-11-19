## Team sets: v1 vs v2

| set | task | n | base | v1 FT no tools | v1 FT + tools | v2 FT no tools | v2 FT + tools | call v1 | call v2 |
|---|---|---|---|---|---|---|---|---|---|
| A_test | track3 | 100 |  42.0 |  33.0 |  96.0 |  34.0 |  98.0 | 100.0 | 100.0 |
| A_test | track5 | 100 |  23.0 |  27.0 |  97.0 |  22.0 |  98.0 | 100.0 | 100.0 |
| A_test | track7 | 100 |  20.0 |  13.0 |  96.0 |  17.0 |  92.0 | 100.0 | 100.0 |
| A_test | order3 | 100 |  46.0 |  35.0 |  81.0 |  34.0 |  74.0 | 100.0 | 100.0 |
| A_test | order5 | 100 |  32.0 |  25.0 |  84.0 |  15.0 |  74.0 | 100.0 | 100.0 |
| A_test | order7 | 100 |  11.0 |  17.0 |  79.0 |  15.0 |  67.0 | 100.0 | 100.0 |
| A_oos2 | track3 | 30 |  33.3 |  40.0 |  80.0 |  33.3 |  70.0 | 100.0 | 100.0 |
| A_oos2 | track5 | 30 |  10.0 |  20.0 |  70.0 |  36.7 |  46.7 | 100.0 | 100.0 |
| A_oos2 | track7 | 30 |  26.7 |  26.7 |  76.7 |  30.0 |  56.7 | 100.0 | 100.0 |
| A_oos2 | order3 | 30 |  23.3 |  16.7 |  73.3 |  30.0 |  63.3 | 100.0 | 100.0 |
| A_oos2 | order5 | 30 |  13.3 |   6.7 |  66.7 |   6.7 |  26.7 | 100.0 | 100.0 |
| A_oos2 | order7 | 30 |  13.3 |  26.7 |  46.7 |  33.3 |  23.3 | 100.0 | 100.0 |
| B_test | temporal | 100 |  15.0 |  87.0 |  96.0 |  74.0 |  98.0 | 100.0 | 100.0 |
| B_test | navigate | 100 |  62.0 |  61.0 |  89.0 |  58.0 |  86.0 | 100.0 | 100.0 |
| B_test | lies | 100 |  55.0 |  61.0 | 100.0 |  79.0 |  91.0 | 100.0 | 100.0 |
| B_oos | meeting | 50 |   0.0 |  10.0 | 100.0 |   8.0 |  78.0 | 100.0 | 100.0 |
| B_oos | robot | 50 |   0.0 |   0.0 |  64.0 |   0.0 |  40.0 | 100.0 | 100.0 |
| B_oos | knights | 50 |  34.0 |  60.0 |  76.0 |  56.0 |  94.0 | 100.0 | 100.0 |
| C_test | multistep_arithmetic_two | 100 |  59.0 |  44.0 |  45.0 |  26.0 |  91.0 |   1.0 |   0.0 |
| C_test | boolean_expressions | 100 |  75.0 |  55.0 |  95.0 |  61.0 |  85.0 |  95.0 |  45.0 |
| C_test | object_counting | 100 |  31.0 |  23.0 |  25.0 |  19.0 |  37.0 |  12.0 |   0.0 |
| C_ood | num_list | 60 |  33.3 /  33.3 |  30.0 /  30.0 |  30.0 /  30.0 |  28.3 /  28.3 |  95.0 /  95.0 |   0.0 |   6.7 |
| C_ood | num_shop | 60 |  73.3 /  90.0 |   3.3 /   3.3 |  45.0 /  45.0 |   3.3 /   3.3 | 100.0 / 100.0 |  48.3 |   0.0 |
| C_ood | bool_code | 60 |  80.0 /  81.7 |  73.3 /  73.3 |  75.0 /  76.7 |  75.0 /  75.0 |  96.7 /  96.7 |  20.0 |   0.0 |
| A_var3 | track5 | 4 |   0.0 |  25.0 |  50.0 |   0.0 |  75.0 | 100.0 | 100.0 |
| A_var3 | trackX | 2 |  50.0 |   0.0 |   0.0 |  50.0 |  50.0 | 100.0 | 100.0 |
| A_var3 | track7 | 1 |   0.0 |   0.0 | 100.0 |   0.0 | 100.0 | 100.0 | 100.0 |
| A_var3 | track3 | 1 |   0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| A_var3 | order5 | 4 |   0.0 |  25.0 |  25.0 |  25.0 |   0.0 |  75.0 | 100.0 |
| A_var3 | order3 | 2 |  50.0 |   0.0 |   0.0 |   0.0 |   0.0 | 100.0 | 100.0 |
| A_var3 | order7 | 2 |  50.0 |   0.0 | 100.0 |   0.0 |  50.0 | 100.0 | 100.0 |
| **A_test** | task mean (raw) | |  29.0 (174/600) |  25.0 (150/600) |  88.8 (533/600) |  22.8 (137/600) |  83.8 (503/600) | | |
| **A_oos2** | task mean (raw) | |  20.0 (36/180) |  22.8 (41/180) |  68.9 (124/180) |  28.3 (51/180) |  47.8 (86/180) | | |
| **A_var3** | task mean (raw) | |  21.4 (3/16) |  21.4 (3/16) |  53.6 (7/16) |  25.0 (3/16) |  53.6 (7/16) | | |
| **B_test** | task mean (raw) | |  44.0 (132/300) |  69.7 (209/300) |  95.0 (285/300) |  70.3 (211/300) |  91.7 (275/300) | | |
| **B_oos** | task mean (raw) | |  11.3 (17/150) |  23.3 (35/150) |  80.0 (120/150) |  21.3 (32/150) |  70.7 (106/150) | | |
| **C_test** | task mean (raw) | |  55.0 (165/300) |  40.7 (122/300) |  55.0 (165/300) |  35.3 (106/300) |  71.0 (213/300) | | |
| **C_ood** | task mean (raw) | |  62.2 (112/180) |  35.6 (64/180) |  50.0 (90/180) |  35.6 (64/180) |  97.2 (175/180) | | |

## Composition test (45 frozen questions): v1 vs v2

| arm | answer strict | lenient | every needed tool used | any A/B call |
|---|---|---|---|---|
| base |   8.9 |   8.9 |   0.0 |   0.0 |
| v1 FT no tools |  17.8 |  17.8 |   0.0 |  97.8 |
| v1 FT + tools |  11.1 |  11.1 |   0.0 |  97.8 |
| v2 FT no tools |  15.6 |  15.6 |  28.9 |  95.6 |
| v2 FT + tools |  24.4 |  26.7 |  24.4 |  95.6 |

Per family, answer strict (v1 FT+tools / v2 FT+tools)
| family | v1 | v2 |
|---|---|---|
| F1_track_expr |   0.0 |  40.0 |
| F2_interval_expr |   0.0 |   0.0 |
| F3_walk_expr |   0.0 |   0.0 |
| F4_truth_expr |   0.0 |   0.0 |
| F5_order_expr |   0.0 |  40.0 |
| F6_track_bool |  40.0 |  40.0 |
| F7_interval_bool |  60.0 |  60.0 |
| F8_interval_expr_calc |   0.0 |   0.0 |
| F9_walk_expr_calc |   0.0 |  40.0 |

Per-tool use on composition (needed / used / correct value). v2 expr/bool/calc = passive: "used" = an `EXPR =` was written
| tool | v1 FT+tools | v2 FT+tools | v2 FT no tools (host-written values) |
|---|---|---|---|
| track | 10/9/2 | 10/8/3 | 10/8/5 |
| order | 5/5/1 | 5/5/5 | 5/5/0 |
| interval | 15/15/15 | 15/15/15 | 15/15/1 |
| walk | 10/10/7 | 10/10/10 | 10/10/0 |
| truth | 5/5/5 | 5/5/5 | 5/5/4 |
| expr | 35/0/0 | 35/10/2 | 35/11/0 |
| bool | 10/0/0 | 10/2/0 | 10/3/1 |
| calc | 10/0/0 | 10/9/2 | 10/9/2 |

## Real BBH, 27 tasks, one run per arm (official prompts and scoring; scores only)

| task | tool task | base | v1 FT no tools | v1 FT + tools | v2 FT no tools | v2 FT + tools | call v1 | call v2 |
|---|---|---|---|---|---|---|---|---|
| boolean_expressions | yes |  82.8 |  58.0 |  96.8 |  66.8 |  80.0 |  99.2 |  55.2 |
| causal_judgement |  |  47.1 |  28.9 |  18.2 |  45.5 |  32.1 | 100.0 |  70.1 |
| date_understanding |  |  54.4 |  42.8 |  39.6 |  38.8 |  14.4 | 100.0 |  98.4 |
| disambiguation_qa |  |  34.0 |  38.0 |  31.6 |  28.0 |  24.0 |  93.2 |  73.6 |
| dyck_languages |  |   0.0 |   0.8 |   0.4 |   1.2 |   1.2 | 100.0 |  60.4 |
| formal_fallacies |  |  52.4 |  42.8 |  16.4 |  43.6 |  15.6 |  94.0 |  94.0 |
| geometric_shapes |  |  28.4 |  20.4 |  22.4 |  18.4 |   3.2 | 100.0 | 100.0 |
| hyperbaton |  |  60.4 |  54.8 |  26.4 |  47.2 |  16.4 |  95.2 |  90.0 |
| logical_deduction_five_objects | yes |  29.2 |  20.4 |  75.6 |  22.0 |  69.6 | 100.0 | 100.0 |
| logical_deduction_seven_objects | yes |  16.4 |  18.8 |  72.8 |  15.2 |  69.2 | 100.0 | 100.0 |
| logical_deduction_three_objects | yes |  57.2 |  37.6 |  76.8 |  30.8 |  73.6 | 100.0 | 100.0 |
| movie_recommendation |  |  35.2 |  19.6 |  15.2 |  26.0 |  26.0 |  94.4 |   3.6 |
| multistep_arithmetic_two | yes |  45.2 |  37.2 |  33.2 |  16.4 | 100.0 |   0.0 |   0.0 |
| navigate | yes |  61.6 |  44.8 |  96.8 |  42.0 |  72.8 | 100.0 | 100.0 |
| object_counting | yes |  66.0 |  42.4 |  42.0 |  39.2 |  57.2 |   8.4 |   9.6 |
| penguins_in_a_table |  |  52.1 |  28.1 |  21.2 |  28.8 |  18.5 |  93.8 |  58.2 |
| reasoning_about_colored_objects |  |  57.2 |  26.4 |  20.4 |  32.8 |  24.8 |  74.0 |  48.0 |
| ruin_names |  |  26.4 |  14.0 |  10.0 |  27.2 |  19.2 |  98.4 |  65.6 |
| salient_translation_error_detection |  |  38.4 |  18.8 |  17.2 |  24.0 |  20.8 | 100.0 |  49.2 |
| snarks |  |  51.7 |  21.3 |  16.9 |  38.8 |  27.5 |  98.3 |  59.0 |
| sports_understanding |  |  46.4 |  46.4 |  46.0 |  46.8 |  46.8 |   0.0 |   0.0 |
| temporal_sequences | yes |  24.8 |  88.8 |  98.8 |  77.6 |  99.6 | 100.0 | 100.0 |
| tracking_shuffled_objects_five_objects | yes |  22.8 |  17.6 | 100.0 |  16.8 |  95.2 | 100.0 | 100.0 |
| tracking_shuffled_objects_seven_objects | yes |  14.8 |  13.6 | 100.0 |  10.0 |  96.4 | 100.0 | 100.0 |
| tracking_shuffled_objects_three_objects | yes |  40.4 |  31.2 | 100.0 |  30.0 |  98.4 | 100.0 | 100.0 |
| web_of_lies | yes |  58.0 |  53.2 | 100.0 |  90.4 | 100.0 | 100.0 | 100.0 |
| word_sorting |  |   3.2 |   0.8 |   0.8 |   3.2 |   3.2 |  95.2 |  20.4 |
| **27-task mean** | | 40.98 | 32.13 | 47.98 | 33.61 | 48.36 |  86.5 |  69.0 |
| **12 tool-task mean** | | 43.27 | 38.63 | 82.73 | 38.10 | 84.33 |  84.0 |  80.4 |
| **15 no-tool-task mean** | | 39.15 | 26.93 | 20.18 | 30.01 | 19.58 |  88.6 |  59.2 |
call rate on the 15 no-tool tasks, v1n:  88.4
call rate on the 15 no-tool tasks, v2n:  59.2
v1t: calls by name on the 15 no-tool tasks (top 8): order 6447, expr 2130, track 901, bool 484, interval 482, truth 443, path 438, movie 256
v2t: calls by name on the 15 no-tool tasks (top 8): track 6442, interval 3939, order 3676, walk 1584, truth 536, path 27, date 16, sort 12

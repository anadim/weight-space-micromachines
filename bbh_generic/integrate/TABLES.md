## Team sets (each team's own prompts/scoring; n per row)

| set | task | n | frozen base | integrated FT, no tools | integrated FT + all 8 tools | call rate (FT+tools) |
|---|---|---|---|---|---|---|
| A_test | track3 | 100 |  42.0 |  33.0 |  96.0 | 100.0 |
| A_test | track5 | 100 |  23.0 |  27.0 |  97.0 | 100.0 |
| A_test | track7 | 100 |  20.0 |  13.0 |  96.0 | 100.0 |
| A_test | order3 | 100 |  46.0 |  35.0 |  81.0 | 100.0 |
| A_test | order5 | 100 |  32.0 |  25.0 |  84.0 | 100.0 |
| A_test | order7 | 100 |  11.0 |  17.0 |  79.0 | 100.0 |
| A_oos2 | track3 | 30 |  33.3 |  40.0 |  80.0 | 100.0 |
| A_oos2 | track5 | 30 |  10.0 |  20.0 |  70.0 | 100.0 |
| A_oos2 | track7 | 30 |  26.7 |  26.7 |  76.7 | 100.0 |
| A_oos2 | order3 | 30 |  23.3 |  16.7 |  73.3 | 100.0 |
| A_oos2 | order5 | 30 |  13.3 |   6.7 |  66.7 | 100.0 |
| A_oos2 | order7 | 30 |  13.3 |  26.7 |  46.7 | 100.0 |
| B_test | temporal | 100 |  15.0 |  87.0 |  96.0 | 100.0 |
| B_test | navigate | 100 |  62.0 |  61.0 |  89.0 | 100.0 |
| B_test | lies | 100 |  55.0 |  61.0 | 100.0 | 100.0 |
| B_oos | meeting | 50 |   0.0 |  10.0 | 100.0 | 100.0 |
| B_oos | robot | 50 |   0.0 |   0.0 |  64.0 | 100.0 |
| B_oos | knights | 50 |  34.0 |  60.0 |  76.0 | 100.0 |
| C_test | multistep_arithmetic_two | 100 |  59.0 |  44.0 |  45.0 |   1.0 |
| C_test | boolean_expressions | 100 |  75.0 |  55.0 |  95.0 |  95.0 |
| C_test | object_counting | 100 |  31.0 |  23.0 |  25.0 |  12.0 |
| C_ood | num_list | 60 |  33.3 /  33.3 |  30.0 /  30.0 |  30.0 /  30.0 |   0.0 |
| C_ood | num_shop | 60 |  73.3 /  90.0 |   3.3 /   3.3 |  45.0 /  45.0 |  48.3 |
| C_ood | bool_code | 60 |  80.0 /  81.7 |  73.3 /  73.3 |  75.0 /  76.7 |  20.0 |
| A_var3 | track5 | 4 |   0.0 |  25.0 |  50.0 | 100.0 |
| A_var3 | trackX | 2 |  50.0 |   0.0 |   0.0 | 100.0 |
| A_var3 | track7 | 1 |   0.0 |   0.0 | 100.0 | 100.0 |
| A_var3 | track3 | 1 |   0.0 | 100.0 | 100.0 | 100.0 |
| A_var3 | order5 | 4 |   0.0 |  25.0 |  25.0 |  75.0 |
| A_var3 | order3 | 2 |  50.0 |   0.0 |   0.0 | 100.0 |
| A_var3 | order7 | 2 |  50.0 |   0.0 | 100.0 | 100.0 |
| A_test | task mean | |  29.0 |  25.0 |  88.8 | |
| A_oos2 | task mean | |  20.0 |  22.8 |  68.9 | |
| A_var3 | task mean | |  21.4 |  21.4 |  53.6 | |
| B_test | task mean | |  44.0 |  69.7 |  95.0 | |
| B_oos | task mean | |  11.3 |  23.3 |  80.0 | |
| C_test | task mean | |  55.0 |  40.7 |  55.0 | |
| C_ood | task mean | |  62.2 |  35.6 |  50.0 | |

### Extra arm: integrated FT + all 8 call tools + Team C PASSIVE tools (expr_gen/calc_gen/bool_gen, fire at "=")

| set | task | n | FT + call tools | FT + call tools + passive C |
|---|---|---|---|---|
| C_test | multistep_arithmetic_two | 100 |  45.0 |  86.0 |
| C_test | boolean_expressions | 100 |  95.0 |  94.0 |
| C_test | object_counting | 100 |  25.0 |  38.0 |
| C_ood | num_list | 60 |  30.0 | 100.0 |
| C_ood | num_shop | 60 |  45.0 | 100.0 |
| C_ood | bool_code | 60 |  75.0 |  96.7 |
| compose | F1_track_expr | 5 |   0.0 |   0.0 |
| compose | F2_interval_expr | 5 |   0.0 |   0.0 |
| compose | F3_walk_expr | 5 |   0.0 |   0.0 |
| compose | F4_truth_expr | 5 |   0.0 |   0.0 |
| compose | F5_order_expr | 5 |   0.0 |   0.0 |
| compose | F6_track_bool | 5 |  40.0 |  60.0 |
| compose | F7_interval_bool | 5 |  60.0 |  60.0 |
| compose | F8_interval_expr_calc | 5 |   0.0 |   0.0 |
| compose | F9_walk_expr_calc | 5 |   0.0 |   0.0 |

## Composition test (45 new questions, 9 families x 5; zero-shot)

| arm | answer acc strict | lenient | called EVERY needed tool | any call |
|---|---|---|---|---|
| frozen base |   8.9 |   8.9 |   0.0 |   0.0 |
| integrated FT, no tools |  17.8 |  17.8 |   0.0 |  97.8 |
| integrated FT + all 8 tools |  11.1 |  11.1 |   0.0 |  97.8 |

Per family (FT + tools): answer strict / lenient / called every needed tool
| family | strict | lenient | every tool |
|---|---|---|---|
| F1_track_expr |   0.0 |   0.0 |   0.0 |
| F2_interval_expr |   0.0 |   0.0 |   0.0 |
| F3_walk_expr |   0.0 |   0.0 |   0.0 |
| F4_truth_expr |   0.0 |   0.0 |   0.0 |
| F5_order_expr |   0.0 |   0.0 |   0.0 |
| F6_track_bool |  40.0 |  40.0 |   0.0 |
| F7_interval_bool |  60.0 |  60.0 |   0.0 |
| F8_interval_expr_calc |   0.0 |   0.0 |   0.0 |
| F9_walk_expr_calc |   0.0 |   0.0 |   0.0 |

Per-tool call correctness (FT + tools; reply = the TOOL's reply; FT no tools: the host writes the reply itself)
| arm | tool | needed | called | non-empty reply | correct reply (canonical numbering) |
|---|---|---|---|---|---|
| ft_notools | bool | 10 | 0 | 0 | 0 |
| ft_notools | calc | 10 | 0 | 0 | 0 |
| ft_notools | expr | 35 | 0 | 0 | 0 |
| ft_notools | interval | 15 | 15 | 15 | 1 |
| ft_notools | order | 5 | 5 | 5 | 1 |
| ft_notools | track | 10 | 9 | 9 | 3 |
| ft_notools | truth | 5 | 5 | 5 | 1 |
| ft_notools | walk | 10 | 10 | 10 | 0 |
| ft_tools | bool | 10 | 0 | 0 | 0 |
| ft_tools | calc | 10 | 0 | 0 | 0 |
| ft_tools | expr | 35 | 0 | 0 | 0 |
| ft_tools | interval | 15 | 15 | 15 | 15 |
| ft_tools | order | 5 | 5 | 2 | 1 |
| ft_tools | track | 10 | 9 | 6 | 2 |
| ft_tools | truth | 5 | 5 | 5 | 5 |
| ft_tools | walk | 10 | 10 | 7 | 7 |

## Real BBH, all 27 tasks (official 3-shot CoT prompts, official exact match; ONE run; scores only)

| task | tool task? | frozen base | integrated FT, no tools | integrated FT + all 8 tools | call rate (FT+tools) |
|---|---|---|---|---|---|
| boolean_expressions | yes |  82.8 |  58.0 |  96.8 |  99.2 |
| causal_judgement |  |  47.1 |  28.9 |  18.2 | 100.0 |
| date_understanding |  |  54.4 |  42.8 |  39.6 | 100.0 |
| disambiguation_qa |  |  34.0 |  38.0 |  31.6 |  93.2 |
| dyck_languages |  |   0.0 |   0.8 |   0.4 | 100.0 |
| formal_fallacies |  |  52.4 |  42.8 |  16.4 |  94.0 |
| geometric_shapes |  |  28.4 |  20.4 |  22.4 | 100.0 |
| hyperbaton |  |  60.4 |  54.8 |  26.4 |  95.2 |
| logical_deduction_five_objects | yes |  29.2 |  20.4 |  75.6 | 100.0 |
| logical_deduction_seven_objects | yes |  16.4 |  18.8 |  72.8 | 100.0 |
| logical_deduction_three_objects | yes |  57.2 |  37.6 |  76.8 | 100.0 |
| movie_recommendation |  |  35.2 |  19.6 |  15.2 |  94.4 |
| multistep_arithmetic_two | yes |  45.2 |  37.2 |  33.2 |   0.0 |
| navigate | yes |  61.6 |  44.8 |  96.8 | 100.0 |
| object_counting | yes |  66.0 |  42.4 |  42.0 |   8.4 |
| penguins_in_a_table |  |  52.1 |  28.1 |  21.2 |  93.8 |
| reasoning_about_colored_objects |  |  57.2 |  26.4 |  20.4 |  74.0 |
| ruin_names |  |  26.4 |  14.0 |  10.0 |  98.4 |
| salient_translation_error_detection |  |  38.4 |  18.8 |  17.2 | 100.0 |
| snarks |  |  51.7 |  21.3 |  16.9 |  98.3 |
| sports_understanding |  |  46.4 |  46.4 |  46.0 |   0.0 |
| temporal_sequences | yes |  24.8 |  88.8 |  98.8 | 100.0 |
| tracking_shuffled_objects_five_objects | yes |  22.8 |  17.6 | 100.0 | 100.0 |
| tracking_shuffled_objects_seven_objects | yes |  14.8 |  13.6 | 100.0 | 100.0 |
| tracking_shuffled_objects_three_objects | yes |  40.4 |  31.2 | 100.0 | 100.0 |
| web_of_lies | yes |  58.0 |  53.2 | 100.0 | 100.0 |
| word_sorting |  |   3.2 |   0.8 |   0.8 |  95.2 |
| **27-task mean** | | 40.98 | 32.13 | 47.98 | |
| **12 tool-task mean** | | 43.27 | 38.63 | 82.73 | |
| **15 no-tool-task mean** | | 39.15 | 26.93 | 20.18 | |

FT+tools call rate on the 15 no-tool tasks:  88.6% of 3511 questions
FT+tools, 12 tool tasks: calls by tool name (non-empty replies): order 1052 (610), track 750 (750), walk 305 (241), interval 259 (249), truth 250 (250), bool 243 (243), expr 95 (1), count 15 (15), enumerate 1 (1)
FT+tools, 15 no-tool tasks: calls by tool name (non-empty replies): order 6447 (50), expr 2130 (64), track 901 (73), bool 484 (7), interval 482 (103), truth 443 (64), path 438 (436), movie 256 (255), walk 190 (6), title 182 (182), arc 144 (143), date 126 (126), name 112 (110), ref 31 (31), kiss 31 (31), translate 29 (27), division 26 (26), walkout 26 (0), list 25 (25), profit 21 (21), speed 21 (21), fix 20 (20), win 18 (18), work 15 (15), priceup 15 (15), fertilizer 13 (13), table 13 (13), similar 11 (11), cut 11 (11), color 8 (8), vote 8 (8), lang 7 (7), plug 7 (7), check 6 (6), sent 6 (6), search 5 (5), cite 5 (5), draw 5 (5), run 5 (5), dist 5 (5), loc 5 (5), block 3 (3), funnel 2 (2), findmovie 2 (2), sim 2 (2), penguin 2 (2), sum 2 (2), time 2 (2), view 2 (2), rating 2 (2), count 2 (2), wave 2 (2), assumption 2 (2), intentional 1 (1), viol 1 (1), p 1 (1), pron 1 (1), match 1 (1), Suf 1 (1), expert 1 (1), arg 1 (1), sorted 1 (1), range 1 (1), bridge 1 (1), rev 1 (1), counting 1 (1), format 1 (1), live 1 (1), place 1 (1), info 1 (1), langtranslate 1 (1), divide 1 (1), coord 1 (1), valtice 1 (0), langsplit 1 (1), distance 1 (1), height 1 (1), fun 1 (1), risk 1 (1)

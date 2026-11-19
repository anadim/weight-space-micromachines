
## GSM8K (n by subset: all 1319, 201+ 1119, clean 1317, 201+ & clean 1117, first 200 200, weird labels 2)

### test / all (n=1319)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 52.2 [49.5, 54.9] | | |
| v2 plain + no rethink (700 tok) | 52.0 [49.3, 54.7] | plain (320 tok) | -0.2 [-1.5, +1.0] |
| v2 plain + rethink always (700 tok) | 56.6 [53.9, 59.3] | v2 plain + no rethink (700 tok) | +4.5 [+3.1, +6.1] |
| v2 plain + rethink gated (detector) (700 tok) | 55.6 [52.9, 58.3] | v2 plain + no rethink (700 tok) | +3.6 [+2.3, +5.1] |
| | | v2 plain + rethink always (700 tok) | -0.9 [-2.0, +0.2] |
| calc (320 tok) | 53.4 [50.6, 56.0] | plain (320 tok) | +1.1 [+0.2, +2.0] |
| v2 calc + no rethink (700 tok) | 53.3 [50.6, 56.0] | v2 plain + no rethink (700 tok) | +1.3 [+0.4, +2.2] |
| v2 calc + rethink always (700 tok) | 57.5 [54.8, 60.2] | v2 plain + no rethink (700 tok) | +5.5 [+3.9, +7.2] |
| | | v2 calc + no rethink (700 tok) | +4.2 [+2.8, +5.8] |
| v2 calc + rethink gated (detector) (700 tok) | 56.7 [54.0, 59.4] | v2 plain + no rethink (700 tok) | +4.7 [+3.1, +6.3] |
| | | v2 calc + rethink always (700 tok) | -0.8 [-1.9, +0.2] |
| q17 plain (320 tok) | 73.1 [70.7, 75.4] | | |
| q17 calc (320 tok) | 72.3 [69.8, 74.7] | q17 plain (320 tok) | -0.8 [-1.6, +0.1] |
| q4 plain (320 tok) | 84.3 [82.3, 86.2] | | |
| q4 calc (320 tok) | 84.7 [82.7, 86.6] | q4 plain (320 tok) | +0.4 [-0.2, +0.9] |

### test / 201+ (n=1119)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 52.3 [49.4, 55.2] | | |
| v2 plain + no rethink (700 tok) | 52.2 [49.2, 55.2] | plain (320 tok) | -0.1 [-1.4, +1.3] |
| v2 plain + rethink always (700 tok) | 56.9 [54.0, 59.9] | v2 plain + no rethink (700 tok) | +4.7 [+3.1, +6.4] |
| v2 plain + rethink gated (detector) (700 tok) | 56.3 [53.4, 59.2] | v2 plain + no rethink (700 tok) | +4.1 [+2.5, +5.7] |
| | | v2 plain + rethink always (700 tok) | -0.6 [-1.8, +0.4] |
| calc (320 tok) | 53.4 [50.5, 56.3] | plain (320 tok) | +1.1 [+0.1, +2.1] |
| v2 calc + no rethink (700 tok) | 53.4 [50.6, 56.4] | v2 plain + no rethink (700 tok) | +1.3 [+0.3, +2.3] |
| v2 calc + rethink always (700 tok) | 57.9 [55.0, 60.9] | v2 plain + no rethink (700 tok) | +5.7 [+3.8, +7.6] |
| | | v2 calc + no rethink (700 tok) | +4.5 [+2.9, +6.2] |
| v2 calc + rethink gated (detector) (700 tok) | 57.4 [54.4, 60.3] | v2 plain + no rethink (700 tok) | +5.2 [+3.4, +7.0] |
| | | v2 calc + rethink always (700 tok) | -0.5 [-1.6, +0.5] |
| q17 plain (320 tok) | 72.6 [69.9, 75.2] | | |
| q17 calc (320 tok) | 71.8 [69.2, 74.5] | q17 plain (320 tok) | -0.7 [-1.6, +0.2] |
| q4 plain (320 tok) | 84.6 [82.5, 86.8] | | |
| q4 calc (320 tok) | 85.2 [83.1, 87.2] | q4 plain (320 tok) | +0.5 [+0.0, +1.2] |

### test / clean (n=1317)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 52.2 [49.6, 55.0] | | |
| v2 plain + no rethink (700 tok) | 52.0 [49.4, 54.7] | plain (320 tok) | -0.2 [-1.5, +1.1] |
| v2 plain + rethink always (700 tok) | 56.6 [53.9, 59.2] | v2 plain + no rethink (700 tok) | +4.6 [+3.1, +6.1] |
| v2 plain + rethink gated (detector) (700 tok) | 55.7 [53.1, 58.5] | v2 plain + no rethink (700 tok) | +3.7 [+2.4, +5.2] |
| | | v2 plain + rethink always (700 tok) | -0.8 [-1.9, +0.2] |
| calc (320 tok) | 53.4 [50.6, 56.1] | plain (320 tok) | +1.1 [+0.2, +2.1] |
| v2 calc + no rethink (700 tok) | 53.3 [50.6, 56.0] | v2 plain + no rethink (700 tok) | +1.3 [+0.4, +2.2] |
| v2 calc + rethink always (700 tok) | 57.6 [55.0, 60.3] | v2 plain + no rethink (700 tok) | +5.6 [+3.9, +7.4] |
| | | v2 calc + no rethink (700 tok) | +4.3 [+2.8, +5.8] |
| v2 calc + rethink gated (detector) (700 tok) | 56.7 [54.0, 59.4] | v2 plain + no rethink (700 tok) | +4.7 [+3.1, +6.4] |
| | | v2 calc + rethink always (700 tok) | -0.9 [-2.0, +0.1] |
| q17 plain (320 tok) | 73.1 [70.8, 75.5] | | |
| q17 calc (320 tok) | 72.4 [69.9, 74.8] | q17 plain (320 tok) | -0.8 [-1.6, +0.1] |
| q4 plain (320 tok) | 84.3 [82.3, 86.3] | | |
| q4 calc (320 tok) | 84.7 [82.7, 86.6] | q4 plain (320 tok) | +0.4 [-0.2, +0.9] |

### test / 201+ & clean (n=1117)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 52.3 [49.4, 55.2] | | |
| v2 plain + no rethink (700 tok) | 52.2 [49.3, 55.1] | plain (320 tok) | -0.1 [-1.4, +1.3] |
| v2 plain + rethink always (700 tok) | 56.9 [54.1, 59.9] | v2 plain + no rethink (700 tok) | +4.7 [+3.1, +6.4] |
| v2 plain + rethink gated (detector) (700 tok) | 56.4 [53.5, 59.4] | v2 plain + no rethink (700 tok) | +4.2 [+2.7, +5.8] |
| | | v2 plain + rethink always (700 tok) | -0.5 [-1.6, +0.5] |
| calc (320 tok) | 53.4 [50.5, 56.3] | plain (320 tok) | +1.1 [+0.1, +2.1] |
| v2 calc + no rethink (700 tok) | 53.4 [50.6, 56.4] | v2 plain + no rethink (700 tok) | +1.3 [+0.3, +2.3] |
| v2 calc + rethink always (700 tok) | 58.0 [55.1, 60.9] | v2 plain + no rethink (700 tok) | +5.8 [+3.9, +7.7] |
| | | v2 calc + no rethink (700 tok) | +4.6 [+3.0, +6.2] |
| v2 calc + rethink gated (detector) (700 tok) | 57.4 [54.5, 60.3] | v2 plain + no rethink (700 tok) | +5.2 [+3.4, +7.0] |
| | | v2 calc + rethink always (700 tok) | -0.6 [-1.7, +0.4] |
| q17 plain (320 tok) | 72.6 [69.9, 75.2] | | |
| q17 calc (320 tok) | 71.9 [69.2, 74.6] | q17 plain (320 tok) | -0.7 [-1.6, +0.2] |
| q4 plain (320 tok) | 84.6 [82.5, 86.7] | | |
| q4 calc (320 tok) | 85.1 [83.2, 87.2] | q4 plain (320 tok) | +0.5 [+0.0, +1.1] |

### test / first 200 (n=200)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 52.0 [45.0, 59.0] | | |
| v2 plain + no rethink (700 tok) | 51.0 [44.0, 58.0] | plain (320 tok) | -1.0 [-4.5, +2.5] |
| v2 plain + rethink always (700 tok) | 54.5 [47.5, 61.5] | v2 plain + no rethink (700 tok) | +3.5 [+0.0, +7.5] |
| v2 plain + rethink gated (detector) (700 tok) | 52.0 [45.0, 59.0] | v2 plain + no rethink (700 tok) | +1.0 [-1.5, +4.0] |
| | | v2 plain + rethink always (700 tok) | -2.5 [-6.0, +0.5] |
| calc (320 tok) | 53.5 [46.5, 60.5] | plain (320 tok) | +1.5 [-0.5, +4.0] |
| v2 calc + no rethink (700 tok) | 52.5 [45.5, 59.5] | v2 plain + no rethink (700 tok) | +1.5 [+0.0, +3.5] |
| v2 calc + rethink always (700 tok) | 55.5 [48.5, 62.5] | v2 plain + no rethink (700 tok) | +4.5 [+0.5, +9.0] |
| | | v2 calc + no rethink (700 tok) | +3.0 [-1.0, +7.0] |
| v2 calc + rethink gated (detector) (700 tok) | 53.0 [46.0, 60.0] | v2 plain + no rethink (700 tok) | +2.0 [-1.5, +5.5] |
| | | v2 calc + rethink always (700 tok) | -2.5 [-6.0, +0.5] |
| q17 plain (320 tok) | 76.0 [70.0, 82.0] | | |
| q17 calc (320 tok) | 75.0 [69.0, 81.0] | q17 plain (320 tok) | -1.0 [-3.0, +1.0] |
| q4 plain (320 tok) | 82.5 [77.0, 87.5] | | |
| q4 calc (320 tok) | 82.0 [76.5, 87.5] | q4 plain (320 tok) | -0.5 [-2.5, +1.0] |

### test / weird labels (n=2)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 50.0 [0.0, 100.0] | | |
| v2 plain + no rethink (700 tok) | 50.0 [0.0, 100.0] | plain (320 tok) | +0.0 [+0.0, +0.0] |
| v2 plain + rethink always (700 tok) | 50.0 [0.0, 100.0] | v2 plain + no rethink (700 tok) | +0.0 [+0.0, +0.0] |
| v2 plain + rethink gated (detector) (700 tok) | 0.0 [0.0, 0.0] | v2 plain + no rethink (700 tok) | -50.0 [-100.0, +0.0] |
| | | v2 plain + rethink always (700 tok) | -50.0 [-100.0, +0.0] |
| calc (320 tok) | 50.0 [0.0, 100.0] | plain (320 tok) | +0.0 [+0.0, +0.0] |
| v2 calc + no rethink (700 tok) | 50.0 [0.0, 100.0] | v2 plain + no rethink (700 tok) | +0.0 [+0.0, +0.0] |
| v2 calc + rethink always (700 tok) | 0.0 [0.0, 0.0] | v2 plain + no rethink (700 tok) | -50.0 [-100.0, +0.0] |
| | | v2 calc + no rethink (700 tok) | -50.0 [-100.0, +0.0] |
| v2 calc + rethink gated (detector) (700 tok) | 50.0 [0.0, 100.0] | v2 plain + no rethink (700 tok) | +0.0 [+0.0, +0.0] |
| | | v2 calc + rethink always (700 tok) | +50.0 [+0.0, +100.0] |
| q17 plain (320 tok) | 50.0 [0.0, 100.0] | | |
| q17 calc (320 tok) | 50.0 [0.0, 100.0] | q17 plain (320 tok) | +0.0 [+0.0, +0.0] |
| q4 plain (320 tok) | 100.0 [100.0, 100.0] | | |
| q4 calc (320 tok) | 100.0 [100.0, 100.0] | q4 plain (320 tok) | +0.0 [+0.0, +0.0] |

## GSM-Hard (n by subset: all 1319, 201+ 1119, clean 901, 201+ & clean 766, first 200 200, weird labels 418)

### hard / all (n=1319)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 27.3 [24.9, 29.7] | | |
| v2 plain + no rethink (700 tok) | 26.6 [24.3, 29.0] | plain (320 tok) | -0.7 [-1.9, +0.5] |
| v2 plain + rethink always (700 tok) | 29.6 [27.1, 32.1] | v2 plain + no rethink (700 tok) | +3.0 [+1.7, +4.2] |
| v2 plain + rethink gated (detector) (700 tok) | 29.0 [26.6, 31.5] | v2 plain + no rethink (700 tok) | +2.4 [+1.3, +3.6] |
| | | v2 plain + rethink always (700 tok) | -0.5 [-1.4, +0.4] |
| calc (320 tok) | 35.0 [32.4, 37.6] | plain (320 tok) | +7.7 [+6.1, +9.4] |
| v2 calc + no rethink (700 tok) | 34.3 [31.8, 36.9] | v2 plain + no rethink (700 tok) | +7.7 [+6.1, +9.4] |
| v2 calc + rethink always (700 tok) | 37.1 [34.5, 39.7] | v2 plain + no rethink (700 tok) | +10.5 [+8.6, +12.5] |
| | | v2 calc + no rethink (700 tok) | +2.8 [+1.6, +4.0] |
| v2 calc + rethink gated (detector) (700 tok) | 35.2 [32.6, 37.8] | v2 plain + no rethink (700 tok) | +8.6 [+6.7, +10.5] |
| | | v2 calc + rethink always (700 tok) | -2.0 [-3.0, -1.0] |
| q17 plain (320 tok) | 45.4 [42.7, 48.1] | | |
| q17 calc (320 tok) | 51.0 [48.3, 53.7] | q17 plain (320 tok) | +5.6 [+4.0, +7.2] |
| q4 plain (320 tok) | 59.0 [56.3, 61.6] | | |
| q4 calc (320 tok) | 60.3 [57.7, 62.9] | q4 plain (320 tok) | +1.4 [+0.2, +2.5] |

### hard / 201+ (n=1119)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 26.2 [23.7, 28.8] | | |
| v2 plain + no rethink (700 tok) | 25.2 [22.7, 27.8] | plain (320 tok) | -1.0 [-2.2, +0.3] |
| v2 plain + rethink always (700 tok) | 28.4 [25.7, 31.0] | v2 plain + no rethink (700 tok) | +3.2 [+1.9, +4.6] |
| v2 plain + rethink gated (detector) (700 tok) | 27.9 [25.3, 30.5] | v2 plain + no rethink (700 tok) | +2.7 [+1.5, +3.9] |
| | | v2 plain + rethink always (700 tok) | -0.5 [-1.5, +0.4] |
| calc (320 tok) | 33.2 [30.4, 35.9] | plain (320 tok) | +7.0 [+5.3, +8.8] |
| v2 calc + no rethink (700 tok) | 32.0 [29.3, 34.8] | v2 plain + no rethink (700 tok) | +6.8 [+5.0, +8.6] |
| v2 calc + rethink always (700 tok) | 35.2 [32.4, 38.0] | v2 plain + no rethink (700 tok) | +10.0 [+8.0, +12.2] |
| | | v2 calc + no rethink (700 tok) | +3.2 [+1.9, +4.6] |
| v2 calc + rethink gated (detector) (700 tok) | 33.1 [30.3, 35.8] | v2 plain + no rethink (700 tok) | +7.9 [+6.0, +9.8] |
| | | v2 calc + rethink always (700 tok) | -2.1 [-3.3, -1.1] |
| q17 plain (320 tok) | 43.9 [41.0, 46.8] | | |
| q17 calc (320 tok) | 48.7 [45.8, 51.7] | q17 plain (320 tok) | +4.8 [+3.3, +6.4] |
| q4 plain (320 tok) | 57.7 [54.9, 60.7] | | |
| q4 calc (320 tok) | 59.1 [56.1, 61.9] | q4 plain (320 tok) | +1.3 [+0.1, +2.6] |

### hard / clean (n=901)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 34.2 [31.1, 37.2] | | |
| v2 plain + no rethink (700 tok) | 34.2 [31.1, 37.2] | plain (320 tok) | +0.0 [-1.4, +1.4] |
| v2 plain + rethink always (700 tok) | 38.0 [34.7, 41.1] | v2 plain + no rethink (700 tok) | +3.8 [+2.1, +5.4] |
| v2 plain + rethink gated (detector) (700 tok) | 37.1 [34.0, 40.2] | v2 plain + no rethink (700 tok) | +2.9 [+1.4, +4.6] |
| | | v2 plain + rethink always (700 tok) | -0.9 [-2.1, +0.3] |
| calc (320 tok) | 41.2 [38.0, 44.4] | plain (320 tok) | +7.0 [+5.1, +8.9] |
| v2 calc + no rethink (700 tok) | 40.5 [37.3, 43.6] | v2 plain + no rethink (700 tok) | +6.3 [+4.4, +8.3] |
| v2 calc + rethink always (700 tok) | 44.2 [40.8, 47.4] | v2 plain + no rethink (700 tok) | +10.0 [+7.7, +12.4] |
| | | v2 calc + no rethink (700 tok) | +3.7 [+2.1, +5.3] |
| v2 calc + rethink gated (detector) (700 tok) | 42.5 [39.3, 45.6] | v2 plain + no rethink (700 tok) | +8.3 [+6.0, +10.7] |
| | | v2 calc + rethink always (700 tok) | -1.7 [-2.9, -0.4] |
| q17 plain (320 tok) | 53.4 [50.2, 56.6] | | |
| q17 calc (320 tok) | 57.9 [54.7, 61.2] | q17 plain (320 tok) | +4.6 [+2.9, +6.3] |
| q4 plain (320 tok) | 67.7 [64.6, 70.7] | | |
| q4 calc (320 tok) | 69.1 [66.1, 72.1] | q4 plain (320 tok) | +1.4 [+0.3, +2.6] |

### hard / 201+ & clean (n=766)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 32.8 [29.5, 36.0] | | |
| v2 plain + no rethink (700 tok) | 32.5 [29.2, 35.8] | plain (320 tok) | -0.3 [-1.7, +1.2] |
| v2 plain + rethink always (700 tok) | 36.6 [33.2, 39.9] | v2 plain + no rethink (700 tok) | +4.0 [+2.2, +5.9] |
| v2 plain + rethink gated (detector) (700 tok) | 35.6 [32.2, 38.9] | v2 plain + no rethink (700 tok) | +3.1 [+1.6, +4.8] |
| | | v2 plain + rethink always (700 tok) | -0.9 [-2.3, +0.4] |
| calc (320 tok) | 38.6 [35.2, 42.2] | plain (320 tok) | +5.9 [+3.9, +7.8] |
| v2 calc + no rethink (700 tok) | 37.3 [33.9, 40.9] | v2 plain + no rethink (700 tok) | +4.8 [+2.9, +6.9] |
| v2 calc + rethink always (700 tok) | 41.4 [38.0, 44.9] | v2 plain + no rethink (700 tok) | +8.9 [+6.4, +11.5] |
| | | v2 calc + no rethink (700 tok) | +4.0 [+2.3, +5.9] |
| v2 calc + rethink gated (detector) (700 tok) | 39.7 [36.3, 43.2] | v2 plain + no rethink (700 tok) | +7.2 [+4.8, +9.5] |
| | | v2 calc + rethink always (700 tok) | -1.7 [-3.0, -0.4] |
| q17 plain (320 tok) | 51.3 [47.8, 54.8] | | |
| q17 calc (320 tok) | 55.5 [52.0, 59.0] | q17 plain (320 tok) | +4.2 [+2.5, +6.0] |
| q4 plain (320 tok) | 66.2 [62.8, 69.5] | | |
| q4 calc (320 tok) | 67.6 [64.4, 70.9] | q4 plain (320 tok) | +1.4 [+0.3, +2.6] |

### hard / first 200 (n=200)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 33.5 [27.0, 40.0] | | |
| v2 plain + no rethink (700 tok) | 34.5 [28.0, 41.0] | plain (320 tok) | +1.0 [-2.5, +4.5] |
| v2 plain + rethink always (700 tok) | 36.0 [29.5, 42.5] | v2 plain + no rethink (700 tok) | +1.5 [-2.0, +5.0] |
| v2 plain + rethink gated (detector) (700 tok) | 35.5 [29.0, 42.0] | v2 plain + no rethink (700 tok) | +1.0 [-2.5, +4.5] |
| | | v2 plain + rethink always (700 tok) | -0.5 [-3.0, +2.0] |
| calc (320 tok) | 45.5 [38.5, 52.5] | plain (320 tok) | +12.0 [+7.0, +17.0] |
| v2 calc + no rethink (700 tok) | 47.5 [40.5, 54.5] | v2 plain + no rethink (700 tok) | +13.0 [+8.5, +18.0] |
| v2 calc + rethink always (700 tok) | 48.0 [41.0, 55.0] | v2 plain + no rethink (700 tok) | +13.5 [+8.0, +19.5] |
| | | v2 calc + no rethink (700 tok) | +0.5 [-2.5, +4.0] |
| v2 calc + rethink gated (detector) (700 tok) | 47.0 [40.0, 54.0] | v2 plain + no rethink (700 tok) | +12.5 [+7.0, +18.5] |
| | | v2 calc + rethink always (700 tok) | -1.0 [-3.5, +1.5] |
| q17 plain (320 tok) | 54.0 [47.0, 61.0] | | |
| q17 calc (320 tok) | 64.0 [57.5, 70.5] | q17 plain (320 tok) | +10.0 [+5.0, +15.0] |
| q4 plain (320 tok) | 66.0 [59.5, 72.5] | | |
| q4 calc (320 tok) | 67.5 [61.0, 74.0] | q4 plain (320 tok) | +1.5 [-1.5, +4.5] |

### hard / weird labels (n=418)
| arm | acc [95% CI] | vs | gain [95% CI] |
|---|---|---|---|
| plain (320 tok) | 12.4 [9.3, 15.8] | | |
| v2 plain + no rethink (700 tok) | 10.3 [7.4, 13.4] | plain (320 tok) | -2.2 [-4.1, -0.2] |
| v2 plain + rethink always (700 tok) | 11.5 [8.6, 14.6] | v2 plain + no rethink (700 tok) | +1.2 [-0.2, +2.6] |
| v2 plain + rethink gated (detector) (700 tok) | 11.7 [8.8, 14.8] | v2 plain + no rethink (700 tok) | +1.4 [+0.2, +2.9] |
| | | v2 plain + rethink always (700 tok) | +0.2 [-0.7, +1.2] |
| calc (320 tok) | 21.8 [17.9, 25.8] | plain (320 tok) | +9.3 [+6.2, +12.7] |
| v2 calc + no rethink (700 tok) | 21.1 [17.2, 25.1] | v2 plain + no rethink (700 tok) | +10.8 [+7.7, +14.1] |
| v2 calc + rethink always (700 tok) | 22.0 [17.9, 26.1] | v2 plain + no rethink (700 tok) | +11.7 [+8.4, +15.1] |
| | | v2 calc + no rethink (700 tok) | +1.0 [-0.5, +2.4] |
| v2 calc + rethink gated (detector) (700 tok) | 19.4 [15.6, 23.2] | v2 plain + no rethink (700 tok) | +9.1 [+6.2, +12.2] |
| | | v2 calc + rethink always (700 tok) | -2.6 [-4.5, -1.0] |
| q17 plain (320 tok) | 28.2 [23.9, 32.5] | | |
| q17 calc (320 tok) | 36.1 [31.6, 40.9] | q17 plain (320 tok) | +7.9 [+4.8, +11.2] |
| q4 plain (320 tok) | 40.2 [35.4, 44.7] | | |
| q4 calc (320 tok) | 41.4 [36.6, 46.2] | q4 plain (320 tok) | +1.2 [-1.4, +3.8] |

## Token budgets and truncation
- test v2 plain + no rethink (700 tok): max_new 700; outputs at the cap 26/1319; >=320 tokens 29; mean tokens 105; no #### answer 37
- test v2 plain + rethink always (700 tok): max_new 700; outputs at the cap 108/1319; >=320 tokens 231; mean tokens 250; no #### answer 37
- test v2 plain + rethink gated (detector) (700 tok): max_new 700; outputs at the cap 66/1319; >=320 tokens 239; mean tokens 213; no #### answer 39
- test v2 calc + no rethink (700 tok): max_new 700; outputs at the cap 24/1319; >=320 tokens 28; mean tokens 105; no #### answer 32
- test v2 calc + rethink always (700 tok): max_new 700; outputs at the cap 104/1319; >=320 tokens 229; mean tokens 250; no #### answer 32
- test v2 calc + rethink gated (detector) (700 tok): max_new 700; outputs at the cap 60/1319; >=320 tokens 246; mean tokens 215; no #### answer 35
- hard v2 plain + no rethink (700 tok): max_new 700; outputs at the cap 61/1319; >=320 tokens 70; mean tokens 144; no #### answer 94
- hard v2 plain + rethink always (700 tok): max_new 700; outputs at the cap 323/1319; >=320 tokens 575; mean tokens 372; no #### answer 94
- hard v2 plain + rethink gated (detector) (700 tok): max_new 700; outputs at the cap 215/1319; >=320 tokens 508; mean tokens 316; no #### answer 86
- hard v2 calc + no rethink (700 tok): max_new 700; outputs at the cap 48/1319; >=320 tokens 60; mean tokens 141; no #### answer 62
- hard v2 calc + rethink always (700 tok): max_new 700; outputs at the cap 348/1319; >=320 tokens 610; mean tokens 382; no #### answer 62
- hard v2 calc + rethink gated (detector) (700 tok): max_new 700; outputs at the cap 225/1319; >=320 tokens 560; mean tokens 329; no #### answer 63
- test plain (320 tok, full text re-tokenized): mean 94 tok; >=316 tok w/o blank-line stop (truncated) 24/1319 (24 of them with no #### answer); no answer overall 36
- test calc (320 tok, full text re-tokenized): mean 95 tok; >=316 tok w/o blank-line stop (truncated) 27/1319 (27 of them with no #### answer); no answer overall 37
- hard plain (320 tok, full text re-tokenized): mean 126 tok; >=316 tok w/o blank-line stop (truncated) 67/1319 (66 of them with no #### answer); no answer overall 95
- hard calc (320 tok, full text re-tokenized): mean 125 tok; >=316 tok w/o blank-line stop (truncated) 56/1319 (55 of them with no #### answer); no answer overall 68

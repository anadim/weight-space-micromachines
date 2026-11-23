dev: 600 items, 36 independent tries per item (dev_sc8_s7.items.jsonl, dev_sc8_s11.items.jsonl, dev_sc8_s13.items.jsonl, dev_sc4_s7.items.jsonl, dev_sc4_s11.items.jsonl, dev_sc4_s13.items.jsonl)
never solved: 23 (3.83%); solved in exactly 1: 5; in <= 2: 36; mean per-try acc 80.74%; majority acc 86.00% (first file)
  never solved in the first  1 tries: 118 (19.67%)
  never solved in the first  4 tries: 52 (8.67%)
  never solved in the first  8 tries: 34 (5.67%)
  never solved in the first 16 tries: 25 (4.17%)
  never solved in the first 24 tries: 24 (4.00%)
  never solved in the first 32 tries: 23 (3.83%)
  never solved in the first 36 tries: 23 (3.83%)
gold labels of the 23 never-solved: {'OK': 17, 'WRONG': 3, 'ILLPOSED': 1, 'AMBIG': 2}; WRONG+AMBIG 5 (22%), +ILLPOSED 6 (26%)
- dev 5 gold 3 | OK | top answers [('1.5', 17), ('2', 5), ('6', 4)]
- dev 6 gold 96 | WRONG | top answers [('576', 36)]
- dev 29 gold 36 | OK | top answers [('200', 17), ('100', 15), ('28', 1)]
- dev 64 gold 118 | OK | top answers [(None, 24), ('20.6', 1), ('7210', 1)]
- dev 107 gold 2200 | OK | top answers [('110', 36)]
- dev 170 gold 360 | OK | top answers [(None, 9), ('210', 5), ('240', 4)]
- dev 213 gold 4 | ILLPOSED | top answers [(None, 21), ('3', 2), ('1.5', 1)]
- dev 215 gold 65 | OK | top answers [('68', 29), ('74', 2), ('60', 1)]
- dev 236 gold 5 | OK | top answers [('9', 34), ('17', 1), ('1', 1)]
- dev 240 gold 28 | OK | top answers [('42', 36)]
- dev 297 gold 320 | OK | top answers [(None, 34), ('297.3', 1), ('2880', 1)]
- dev 303 gold 90 | WRONG | top answers [('180', 36)]
- dev 361 gold 10 | OK | top answers [('4', 27), ('20', 2), ('17', 2)]
- dev 374 gold 25600000 | OK | top answers [('28800000', 6), (None, 4), ('256000000', 3)]
- dev 386 gold 3 | OK | top answers [('6', 36)]
- dev 432 gold 35 | OK | top answers [('45', 36)]
- dev 445 gold 66 | OK | top answers [('43', 18), ('89', 9), ('83', 9)]
- dev 488 gold 7 | OK | top answers [('13', 36)]
- dev 496 gold 84 | OK | top answers [('32', 18), ('80', 7), ('48', 4)]
- dev 519 gold 4 | WRONG | top answers [('5', 20), ('10', 16)]
- dev 522 gold 28 | OK | top answers [('47', 23), ('27', 3), ('49', 2)]
- dev 564 gold 40 | AMBIG | top answers [('30', 34), ('10', 2)]
- dev 567 gold 350 | AMBIG | top answers [('3.5', 17), ('3.50', 9), ('2.30', 2)]

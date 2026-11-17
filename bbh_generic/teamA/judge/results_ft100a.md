### round(s) 3
| id | target | host+tool | host call -> reply | tool reply = gold? | FT no tool | base | gold call | ref | tool alone |
|---|---|---|---|---|---|---|---|---|---|
| M1_zoo5 | (C) | (A) X | `track 12345:24,15,3=6,52?1` -> `5` | WRONG value | (A) X | (A) X | `track 12345:24,15,3=6,52@1` | `2` | '2' |
| M2_teams5 | (D) | (D) OK | `track 12345:24,3=6,1<2,53?1` -> `4` | yes | (E) X | (A) X | `track 12345:24,3=6,1<2,53?1` | `4` | '4' |
| M3_words5 | (D) | (C) X | `order 5:14,25,31,45?3` -> `` | empty (fail-safe) | (D) OK | (B) X | `track 12345:14,25,31,45?3` | `4` | '4' |
| M4_array4 | (C) | (A) X | `track 4816:13,21,21,21?2` -> `1` | WRONG value | (C) OK | - X | `track 4816:14,2<3,31?3` | `6` | '6' |
| M5_scarves7 | (E) | (D) X | `track 1234567:27,31,54,52,73,14?2` -> `4` | WRONG value | (A) X | (F) X | `track 1234567:27,31,64,52,56,73,14?2` | `5` | '5' |
| M6_seats5 | (E) | (E) OK | `track 12345:25,13,42,35@2` -> `3` | yes | (D) X | (D) X | `track 12345:25,13,42,35@2` | `3` | '3' |
| M7_souvenirs3 | (A) | (A) OK | `track 123:23,12,32?3` -> `1` | yes | (B) X | (B) X | `track 123:23,21,32?3` | `1` | '1' |
| M8_cells4 | (B) | (D) X | (no call) | no call | (D) X | (B) OK | `track 3795:24,3<1,12,4=2?2` | `3` | '3' |
| judge3_order_1 | (A) | (E) X | `order 5:4=-1,3>5,1+3,2!=1,2>5?1` -> `5` | WRONG value | (D) X | (D) X | `order 5:4=1,5<3,3+1,2!=-1,5<2?-1` | `1` | '1' |
| judge3_order_2 | (D) | (C) X | `order 5:1<2,4=5,3+5,2!=2,1+3?2` -> `3` | WRONG value | (A) X | - X | `order 5:1<2,4=5,5+3,2!=2,3+1@2` | `4` | '4' |
| judge3_order_3 | (C) | (A) X | (no call) | no call | (A) X | (C) OK | `order 3:2=-1,1<3?1` | `1` | '1' |
| judge3_order_4 | (D) | (D) OK | `order 7:6<4,1=3,3=-1,2=1,5<1,7=-3,4>7?-2` -> `4` | yes | (A) X | (D) OK | `order 7:6<4,1=3,3=-1,2=1,5<1,7=-3,4>7?-2` | `4` | '4' |
| judge3_order_5 | (B) | (E) X | `order 5:5=1,2+3,4<3,1!=5,4+5?1` -> `` | empty (fail-safe) | (B) OK | (A) X | `order 5:5=1,3+2,4<3,1!=-1,5+4?-1` | `2` | '2' |
| judge3_order_6 | (B) | (C) X | `order 5:3>5,4+1,4<1,2!=1,1=2?4` -> `` | empty (fail-safe) | (A) X | - X | `order 5:5<3,4+1,2=-1,1=2?4` | `3` | '3' |
| judge3_order_7 | (C) | (C) OK | `order 3:3!=1,2>3?middle` -> `` | empty (fail-safe) | (C) OK | (B) X | `order 3:3!=1,2>3?2` | `3` | '3' |
| judge3_order_8 | (E) | (B) X | `order 7:6=-1,1=1,4+3,2<3,5+7,7+4?-4` -> `7` | WRONG value | (F) X | - X | `order 7:6=-1,1=1,3+4,2<3,7<5,4+7?-4` | `4` | '4' |

tool reply on the host call: {'WRONG value': 6, 'yes': 4, 'empty (fail-safe)': 4, 'no call': 2}
host+tool 5/16, FT no tool 4/16, base 3/16, tool alone exact 16/16
### round(s) 12
| id | target | host+tool | host call -> reply | tool reply = gold? | FT no tool | base | gold call | ref | tool alone |
|---|---|---|---|---|---|---|---|---|---|
| J1_mugs3 | (B) | (B) OK | `track 123:31,23,12?3` -> `2` | yes | (C) X | (A) X | `track 123:31,23,12?3` | `2` | '2' |
| J2_parking5 | (D) | (D) OK | `track 12345:25,13,42,51,34@4` -> `2` | yes | (A) X | (A) X | `track 12345:25,13,42,51,34@4` | `2` | '2' |
| J3_cards3 | (D) | (D) OK | `track 123:23,1=4,12?2` -> `4` | yes | (D) OK | (D) OK | `track 123:23,1=4,12?2` | `4` | '4' |
| J4_code3 | (B) | (B) OK | `track 725:13,21,13?2` -> `5` | yes | (B) OK | (B) OK | `track 725:13,2<1,13?2` | `5` | '5' |
| J5_books7 | (F) | (E) X | `track 1234567:51,73,26,45,32,17,64?3` -> `6` | yes | (A) X | (E) X | `track 1234567:51,73,26,45,32,17,64?3` | `6` | '6' |
| J6_umbrellas5 | (D) | (B) X | `track 12345:31,42,53,21,45?1` -> `4` | WRONG value | (A) X | (C) X | `track 12345:31,42,53,21,45?2` | `3` | '3' |
| J7_cups3 | (A) | (B) X | `track 123:123,132,213@2` -> `` | empty (fail-safe) | (B) X | (C) X | `track 123:13,21,32@2` | `1` | '1' |
| J8_docks5 | (C) | (D) X | `track 12345:24,1=6,15,32,3=7@4` -> `` | empty (fail-safe) | (A) X | (D) X | `track 12345:24,1=6,15,32@4` | `3` | '3' |
| J9_scorebooks7 | (G) | (G) OK | `track 1234567:27,31,64,52,73,45,16,23,75?4` -> `7` | yes | (A) X | (E) X | `track 1234567:27,31,64,52,73,45,16,23,75?4` | `7` | '7' |
| J10_jerseys4 | (B) | (B) OK | `track 1234:14,21,32?1` -> `2` | WRONG value | (A) X | (B) OK | `track 8361:14,21,32?1` | `3` | '3' |
| judge_order_1 | (B) | (B) OK | `order 3:3<1,2<3?1` -> `2` | yes | (C) X | - X | `order 3:3<1,2<3?1` | `2` | '2' |
| judge_order_2 | (D) | (D) OK | `order 5:3+1,5=4,4>5,2>1?2` -> `1` | yes | (B) X | (B) X | `order 5:1+3,5=4,4=-1,2<1?2` | `1` | '1' |
| judge_order_3 | (F) | (F) OK | `order 7:1=-1,6=-1,7+4,3+7,5>2,2>3?5` -> `` | empty (fail-safe) | (G) X | (A) X | `order 7:1=1,6=-1,4+7,3~7,2<5,3<2?5` | `2` | '2' |
| judge_order_4 | (C) | (C) OK | `order 5:3<2,5=-1,1+4,4<3@3` -> `3` | yes | (C) OK | (D) X | `order 5:3<2,5=5,1+4,4<3@3` | `3` | '3' |
| judge_order_5 | (B) | (B) OK | `order 3:1=1,2=-1?-1` -> `2` | yes | (C) X | (A) X | `order 3:2>3,1=1?-1` | `2` | '2' |
| judge_order_6 | (D) | (A) X | `order 5:3<1,5=-1,2>3,4!=1,4>3?-2` -> `` | empty (fail-safe) | (C) X | - X | `order 5:3<1,5=-1,2<3,4!=1,4<3?-2` | `1` | '1' |
| judge_order_7 | (C) | (C) OK | `order 7:5=1,3=-1,4=-4,1+3,2>7,6<4?2` -> `` | empty (fail-safe) | (A) X | (C) OK | `order 7:5=-1,3=1,4=4,3+1,7<2,6<4?-2` | `2` | '2' |
| judge_order_8 | (C) | (C) OK | `order 3:2>3,1=2?1` -> `3` | yes | (B) X | (A) X | `order 3:2>3,1=2?1` | `3` | '3' |
| judge_order_9 | (D) | (D) OK | `order 5:3=1,4=-1,1+5,2>1,1=2?4` -> `2` | yes | (C) X | (A) X | `order 5:3=1,4=-1,1<5,5<2,1<2,1=2?4` | `2` | '2' |
| judge_order_10 | (B) | (A) X | (no call) | no call | (A) X | (B) OK | `None` | `` |  |
| K1_hotel5 | (D) | (A) X | `track 12345:34,1=6,25,2015@1` -> `` | empty (fail-safe) | (B) X | (A) X | `track 12345:34,1=6,25,13?1` | `4` | '4' |
| K2_relay5 | (D) | (D) OK | `track 12345:25,3=5,14,23?1` -> `4` | yes | (B) X | (B) X | `track 12345:25,3=6,14,23@1` | `4` | '4' |
| K3_bikes7 | (E) | (E) OK | `track 1234567:62,71,46,37,54,23,15?4` -> `5` | yes | (A) X | (D) X | `track 1234567:62,71,46,37,54,23,15?4` | `5` | '5' |
| K4_bedrooms5 | (C) | (C) OK | `track 12345:25,42,42,15?5` -> `1` | yes | (E) X | (C) OK | `track 12345:25,42,24,15?5` | `1` | '1' |
| K5_phones3 | (C) | (C) OK | `track 123:12,23,31@2` -> `3` | yes | (B) X | (A) X | `track 123:12,23,13?2` | `3` | '3' |
| K6_sims4 | (D) | (C) X | `track 1234:23,1=5,34,13?2` -> `3` | WRONG value | (B) X | (C) X | `track 1234:23,1=5,34,13@2` | `4` | '4' |
| K7_chores6 | (D) | (B) X | `track 123456:51,62,15,35,46,34,25?1` -> `1` | WRONG value | (A) X | (B) X | `track 123456:51,62,35,46,34,25@1` | `4` | '4' |
| K8_wallpapers4 | (C) | (C) OK | `track 1234:13,4<1,24?2` -> `3` | yes | (B) X | (D) X | `track 1234:13,4<1,24?2` | `3` | '3' |
| judge2_order_1 | (D) | (D) OK | `order 5:1<3,4=-1,2+3,5=2?3` -> `2` | yes | (E) X | (B) X | `order 5:1<3,4=-1,2+3,5=2?3` | `2` | '2' |
| judge2_order_2 | (C) | (B) X | `track 12345:5=1,3+2,4+3,1!=5,1+5@2` -> `` | empty (fail-safe) | (A) X | (A) X | `order 5:5=1,2<3,4+3,1!=-1,5+1@2` | `3` | '3' |
| judge2_order_3 | (D) | (D) OK | `order 7:3<6,7=1,2=3,4>5,1=-1,6<5?-2` -> `4` | yes | (C) X | (D) OK | `order 7:3<6,7=1,2=3,4>5,1=-1,6<5?-2` | `4` | '4' |
| judge2_order_4 | (A) | (C) X | (no call) | no call | (C) X | (C) X | `order 3:2<3,1>3?-1` | `1` | '1' |
| judge2_order_5 | (A) | (B) X | `order 3:2!=1,3+1?3` -> `2` | WRONG value | (B) X | (C) X | `order 3:2!=1,1+3` | `132` | '132' |
| judge2_order_6 | (C) | (C) OK | `order 7:4+2,3=1,7=-1,1<2,6+4,5!=2,5<6?5` -> `4` | yes | (G) X | (B) X | `order 7:2+4,3=1,7=-1,1<2,4+6,5!=2,5<6?5` | `4` | '4' |
| judge2_order_7 | (D) | (B) X | `order 5:1=1,2>5,3<4,2=2?2` -> `` | empty (fail-safe) | (B) X | - X | `order 5:1=-1,2>5,3<4,2=2?-2` | `4` | '4' |
| judge2_order_8 | (C) | (C) OK | `order 3:1!=1,2>1,2>3?2` -> `1` | yes | (A) X | (A) X | `order 3:1!=1,2>1,2>3?2` | `1` | '1' |

tool reply on the host call: {'yes': 21, 'WRONG value': 5, 'empty (fail-safe)': 8, 'no call': 2}
host+tool 23/36, FT no tool 3/36, base 7/36, tool alone exact 35/35

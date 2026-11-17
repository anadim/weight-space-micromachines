| set | host | track3 | track5 | track7 | order3 | order5 | order7 |
|---|---|---|---|---|---|---|---|
| dev | frozen base | 36.7 | 30.0 | 20.0 | 53.3 | 30.0 | 3.3 |
| dev | FT100 no tools | 40.0 | 26.7 | 16.7 | 36.7 | 13.3 | 16.7 |
| dev | FT100 + tools | 100.0 | 90.0 | 93.3 | 86.7 | 90.0 | 90.0 |
| test | frozen base | 39.0 | 22.0 | 19.0 | 46.0 | 27.0 | 18.0 |
| test | FT100 no tools | 25.0 | 20.0 | 13.0 | 33.0 | 20.0 | 14.0 |
| test | FT100 + tools | 92.0 | 91.0 | 88.0 | 84.0 | 84.0 | 72.0 |
| test | FT100 age/price-ablated + tools | 92.0 | 97.0 | 97.0 | 84.0 | 90.0 | 81.0 |
| oos2 | frozen base | 40.0 | 10.0 | 23.3 | 33.3 | 13.3 | 16.7 |
| oos2 | FT100 no tools | 40.0 | 30.0 | 20.0 | 30.0 | 13.3 | 23.3 |
| oos2 | FT100 + tools | 83.3 | 56.7 | 63.3 | 70.0 | 50.0 | 36.7 |
| oos2 | FT100 age/price-ablated + tools | 73.3 | 36.7 | 56.7 | 56.7 | 50.0 | 63.3 |
| oos3 | frozen base | 35.0 | 20.0 | 25.0 | 45.0 | 35.0 | 25.0 |
| oos3 | FT100 no tools | 20.0 | 25.0 | 10.0 | 45.0 | 15.0 | 20.0 |
| oos3 | FT100 + tools | 70.0 | 65.0 | 75.0 | 80.0 | 65.0 | 55.0 |
| oos | frozen base | 33.3 | 23.3 | 6.7 | 40.0 | 20.0 | 16.7 |
| oos | FT100 no tools | 40.0 | 10.0 | 6.7 | 36.7 | 23.3 | 13.3 |
| oos | FT100 + tools | 60.0 | 50.0 | 50.0 | 73.3 | 50.0 | 53.3 |
== test ft100a + tools
| task | n | answer acc | tool reply = gold | empty reply (fail-safe) | wrong value | no call |
|---|---|---|---|---|---|---|
| track3 | 100 | 92.0 | 92 | 0 | 8 | 0 |
| track5 | 100 | 91.0 | 93 | 1 | 6 | 0 |
| track7 | 100 | 88.0 | 90 | 0 | 10 | 0 |
| order3 | 100 | 84.0 | 80 | 8 | 12 | 0 |
| order5 | 100 | 84.0 | 81 | 12 | 7 | 0 |
| order7 | 100 | 72.0 | 69 | 25 | 6 | 0 |
== oos2 ft100a + tools
| task | n | answer acc | tool reply = gold | empty reply (fail-safe) | wrong value | no call |
|---|---|---|---|---|---|---|
| track3 | 30 | 83.3 | 24 | 3 | 3 | 0 |
| track5 | 30 | 56.7 | 16 | 8 | 6 | 0 |
| track7 | 30 | 63.3 | 20 | 1 | 9 | 0 |
| order3 | 30 | 70.0 | 21 | 4 | 5 | 0 |
| order5 | 30 | 50.0 | 12 | 13 | 5 | 0 |
| order7 | 30 | 36.7 | 9 | 17 | 4 | 0 |
== oos3 ft100a + tools
| task | n | answer acc | tool reply = gold | empty reply (fail-safe) | wrong value | no call |
|---|---|---|---|---|---|---|
| track3 | 20 | 70.0 | 10 | 8 | 2 | 0 |
| track5 | 20 | 65.0 | 11 | 6 | 3 | 0 |
| track7 | 20 | 75.0 | 11 | 8 | 1 | 0 |
| order3 | 20 | 80.0 | 16 | 0 | 4 | 0 |
| order5 | 20 | 65.0 | 12 | 5 | 3 | 0 |
| order7 | 20 | 55.0 | 10 | 8 | 2 | 0 |
== test ablation
| task | n | answer acc | tool reply = gold | empty reply (fail-safe) | wrong value | no call |
|---|---|---|---|---|---|---|
| track3 | 100 | 92.0 | 92 | 2 | 6 | 0 |
| track5 | 100 | 97.0 | 97 | 0 | 3 | 0 |
| track7 | 100 | 97.0 | 96 | 2 | 2 | 0 |
| order3 | 100 | 84.0 | 85 | 3 | 9 | 3 |
| order5 | 100 | 90.0 | 86 | 9 | 3 | 2 |
| order7 | 100 | 81.0 | 78 | 19 | 3 | 0 |

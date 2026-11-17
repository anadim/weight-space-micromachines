# Round 2 (all 14 tools)
The circuit changed after round 1: NOTES.md "Change log, part 6" (read it, and re-read the CURRENT circuit.py / tables.py for your
tool). Your round-1 items are now SEEN (part 5/6 were shaped by them), so round 2 needs FRESH items:
1. Write >= 12 NEW items for your tool to ifeval/heldout/judge/{TOOL}_variants_r2.jsonl, keys {KEYBASE}+100+i, same JSON
   format as round 1 (fields key, prompt, optional system/history, instruction_id_list, kwargs, note). They must differ from your
   round-1 items and from gen_data.py (new wordings, new topics, new settings), >= 7 positives and >= 5 negatives (mentions without
   a request). Include the specific risks you flagged in round 1 that the fixes claim to address, plus anything new you see in the
   current code. Keep your expected outcome in "note". Do not open fresh/ood/IFEval data.
2. Report back (variants path + any code findings on the current code). I will run them and send results; then you write
   ifeval/heldout/JUDGE_{TOOL}_2.md with the final verdict (same criteria as round 1: no reward hacking / Python logic,
   correct or fail-safe on positives, near-zero false fires on negatives, USEFUL-GENERALLY).

<!-- Copy of the protocol given to the BBH-redo agents (Teams A, B, C, the judges, the eval and integration agents), as written
     during the project; only rule 7 (machine logistics) was reworded for this release. -->
# BBH redo: held-out protocol (binding for every agent)

Goal: the nine BBH implants must be GENERIC procedures, developed only on our own data, and evaluated ONCE on real BBH.
Context: the original implants were debugged on BBH test questions themselves, and three (deduction engine, swap tracker,
schedule reader) hard-code BBH's vocabulary/phrasing (e.g. a closed list of BBH's 50 object names; the 7 names Alice..Gertrude).
The author calls that reward hacking. This redo removes it.

## Hard rules
1. DO NOT open, grep, print, or load any BBH test data: anything under */bbh/data/*, bbh test json files, results/bbh/*.json
   (per-question outputs), old failure logs, or old per-question traces. Do not ask other agents for them.
   You MAY read BBH's official few-shot prompt files (the 3-shot CoT prompts, e.g. */bbh/cot-prompts/*.txt or wherever the
   repo keeps them): the prompts are part of the input at test time, not test data. You may also use the public task
   descriptions (what each task is about, answer format).
2. Build your OWN generators per task. Each generator must produce questions that look like the task but vary:
   - names/objects/places NOT drawn from BBH's vocabulary (use your own lists: e.g. random first names from many cultures,
     everyday objects, colors, animals, etc.), with BBH-like surface form (same answer format and option style);
   - several phrasings per sentence type (paraphrases of instructions / clue templates), not just one template;
   - difficulty spread matching BBH (e.g. 3/5/7 objects, chain lengths, number ranges).
   Use seed 0 for DEV (30 questions per task, the author asked for ~30) and seed 1 for a FRESH TEST set (100 per task,
   with extra paraphrase variety you did not tune on). Never look at fresh-test failures while developing.
3. Implants must be fully compiled into weights (no Python logic at inference; see the repo's machine.py/hdl.py pattern).
   They must be GENERIC: no closed lists of BBH entities; readers keyed on structure (token classes, punctuation, generic
   relation words), not on BBH-specific strings. If an implant needs a word list (e.g. "left/right/before/after"),
   it must be a generic linguistic list, written down and justified, not mined from BBH.
4. Develop only on DEV (seed 0) with the frozen host Qwen3-0.6B-Base and the official BBH 3-shot prompt for that task
   (prompt + your generated question). Iterate the loop: run, read DEV failures, fix the implant, repeat.
   Keep a NOTEBOOK.md with every change and the DEV failure that motivated it.
5. Ask yourself after every change: "is this fitting to the test?" If a fix only helps because of a quirk of BBH wording
   you remember from before, don't make it.
6. When done: freeze. Report DEV accuracy and FRESH TEST accuracy (seed 1) for plain vs with-implants. Do NOT run real BBH.
   A separate eval agent runs real BBH once, after an auditor checks the circuits.
7. Code goes in this repository in a new folder bbh_generic/ (don't overwrite existing implants; new versions get new
   names, e.g. track_gen, logic_gen, time_gen). Model runs happen on the GPU machines, each team in its own working
   directory. No git commits/pushes.

## Tasks (12) and implant ownership
- Team A: tracking_shuffled_objects (3/5/7) -> swap tracker; logical_deduction (3/5/7) -> deduction engine.
- Team B: temporal_sequences -> schedule reader; navigate -> navigator; web_of_lies -> truth tracker.
- Team C: multistep_arithmetic_two -> calculator/sum/expression; boolean_expressions -> boolean; object_counting -> sum.

## Added by the author (binding)
8. GENERIC TOOLS, as generic as the calculator. Do not build "BBH task solvers". Build general-purpose tools that a model could
   use on any text, e.g.:
   - calculator (exists), expression evaluator, boolean evaluator;
   - a STATE / PERMUTATION tracker (assign X->v, swap(X,Y), query X) -- replaces the "swap tracker";
   - an ORDERING CONSTRAINT solver (items + pairwise/positional constraints -> the consistent order, query position) -- replaces
     the "deduction engine";
   - an INTERVAL tool (busy intervals + open/close bounds -> free interval) -- replaces the "schedule reader";
   - a 2-D position/heading accumulator (moves/turns -> position, at-origin?) -- navigator;
   - a truth-propagation tool (A tells truth / X says Y lies -> truth of X) -- truth tracker.
   Each tool takes ARGUMENTS written in text by the model (like `<tool_call>48 / 2</tool_call>`), not hard-coded BBH phrasing.
9. Two allowed ways to get the model to use the tools:
   (a) frozen model + few-shot (the official BBH 3-shot prompt is fixed; you may NOT edit it) with implants that trigger on
       generic patterns in the model's own reasoning; or
   (b) FINE-TUNE the model (full or LoRA) to emit tool calls in a fixed call format, trained ONLY on our own generated data
       (seed 2 = train split, any size; never BBH, never dev/fresh-test). Then evaluate with the official BBH prompts
       unchanged. Option (b) is explicitly OK with the author and is probably cleaner. Disclose which you used.
10. JUDGE EVERY IMPLANT. After you build (or substantially change) each tool, spawn an independent judge subagent (Agent tool,
    general-purpose) with ONLY: the tool's code, its generator code, the protocol, and your NOTEBOOK entry. Brief: "Act as a
    skeptical reviewer looking for reward hacking / test fitting: closed vocabularies, BBH-specific strings or templates,
    behavior that only works on BBH wording, Python logic at inference, leakage from BBH data, fixes justified only by memory of
    BBH. Verdict PASS/FAIL with concrete reasons." Save each verdict to JUDGE_<tool>_<n>.md. A tool is usable only after PASS.
    Fix and re-judge on FAIL.
11. JUDGE QUESTION (added by the author, mandatory): every judge must ask "Can this tool be used somewhere that looks a bit
    different?" Concretely, the judge writes at least 5 REWORDED variants of the task input (paraphrased instructions,
    different names/objects/numbers, different sentence order, synonyms, an extra irrelevant sentence) that the dev generator
    would NOT produce, and the team runs the tool on them (host + tool, and the tool alone on the relevant text). A PASS needs
    the tool to still fire correctly on most variants (or fail safe by staying silent). Record the variants and results in
    the JUDGE_*.md file.
12. NOT ONLY ONE STYLE OF QUESTION (author). A tool must be as generic as the calculator or the eyes: usable outside the task it
    was built for. Each team must show every tool working in at least one DIFFERENT setting than its benchmark style, e.g.:
    state tracker on an inventory/recipe/card-game story; ordering solver on a race/ranking/seating story; interval tool on a
    meeting calendar; position accumulator on a robot/delivery route; truth propagation on a "who said what" story with
    different wording; boolean/expression tools inside ordinary prose or code-like text. Report these out-of-style results next
    to DEV/FRESH. A tool that only works on its benchmark's question style FAILS the judge.
13. "IS f(x) GENERALLY USEFUL?" IS PART OF THE LOOP (author). Before writing any tool, and again at judging, answer in the
    NOTEBOOK and the JUDGE file: what is the function f(x) this tool computes, stated without reference to the benchmark,
    and would it be generally useful to have in a model (name 3 settings outside the benchmark where it helps)?
    - If YES: keep the tool, and make its interface general enough to serve those settings.
    - If NO (it only makes sense for this benchmark's puzzle): do not build it, or drop it, even if it would raise the score.
    The judge must give this verdict explicitly: USEFUL-GENERALLY yes/no, with the 3 settings.
14. ONE MODEL, ALL TOOLS, COMPOSITION (author). Fine-tuning the host to call the tools is fine whenever few-shot prompting
    doesn't work out of the box (train only on our own generated data). All BBH teams use ONE shared call envelope so a single
    model can call every tool:  <tool_call>NAME ARGS</tool_call><tool_response>RESULT</tool_response>
    (host writes up to </tool_call>; the tool writes the response; empty RESULT = error/undetermined). Team A's
    bbh_generic/teamA/callfmt.py is the reference (tools `track`, `order`). Team B: `interval`, `walk`, `truth`; Team C:
    `calc`, `expr`, `bool` (and `count` if kept). Each team documents its NAME ARGS grammar in a callfmt-style docstring.
    At the end an integration step attaches ALL tools to one model (fine-tuned on single-tool generated data only) and tests
    it on NEW composed questions that need several tools at once, of a kind never seen in dev/test/train (e.g. a story with
    swaps, then an ordering clue, then arithmetic on the result). Report whether the model calls all the needed tools and
    gets the answer.

15. SMALL FINE-TUNES (author, 2026-11-17). Host fine-tunes for tool calling use about 100 examples (rows), not thousands.
    Big fine-tunes (12k-29k rows) teach the task itself and hide what the tools add. Always report base / FT-no-tools / FT+tools.
    The integration fine-tune (all tools, one model) also uses a small set (~100 per team, a few hundred total).

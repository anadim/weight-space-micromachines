# [26] host decision

**Host for all three benchmarks: Qwen/Qwen3-0.6B-Base.**

Reasons:
- GSM8K: every implant, rung and RL arm (P, CW, A1W) is already on Base. Changing the host would mean refitting and rerunning all of them.
- BBH dev check (408 items, 12 tool families):
  - Base scores 36.52, about the same as Qwen3-0.6B (38.97).
  - Neither host ever emits `<tool_call>`, so on both of them the 5 CALL tools are inert.
  - So moving to Qwen3-0.6B buys nothing on BBH.
- IFEval: the 14-rule circuit was already attached to Base in the library runs. Its readers/writers are refit to Base where needed, and idle bit-identity is re-verified.
  - Prompt format: the chat template if the dev check (j23634) shows Base follows it; otherwise the plain-text format. Whichever is chosen is fixed for all 4 rows.

BBH rows 3/4, call fine-tune counted as part of the implant install (disclosed):
- Install = the §6.1 call-FT recipe, run on Base:
  - bbh_generic/teamA/ft.py on the teams' own train_union_v2: 432 rows, lr 2e-5, 6 epochs, seed 0, no BBH test data;
  - then the 8 tools;
  - then the call gate, refit on exactly that host with the original recipe on own data.
- This is a weight change of the whole host, not a frozen add-on. Table rows:
  - 1 = Base;
  - 2 = Base + RL;
  - 3 = Base + call-FT + 8 tools + gate;
  - 4 = row 3 + RL with the tools attached;
  - control row 3c = Base + call-FT, tools off. It separates the effect of the fine-tune from the effect of the tools.
- RL for row 4 starts from the call-FT'd Base. Row 2 starts from plain Base. Both use the same recipe and the same train data.

RL recipe (rows 2 and 4, every task):
- GRPO (Dr.GRPO), 300 updates, 64 prompts × G8, lr 1e-6, T = 1, seeds 0/1/2, train data only.
- Dev checks every 25 updates, never test.
- One test evaluation at u300 (BBH-27 + the 12 tool tasks; IFEval 541 strict prompt-level), with the same prompt format, decoding and token budget in every row. Report tok/q.

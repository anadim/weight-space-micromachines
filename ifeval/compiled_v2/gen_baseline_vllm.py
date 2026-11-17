"""vLLM greedy baseline answers of the host (Qwen3-0.6B, thinking off) to all 541 IFEval prompts. Used ONLY to collect the
states that solve_dirs.py fits the write vectors (dirs.pt) on; solve_dirs.py reads the EVEN keys of this file only.

cd ifeval/compiled_v2 && python gen_baseline_vllm.py Qwen/Qwen3-0.6B ../data/q3_nothink.jsonl nothink
(vLLM, bf16, temperature 0, max_tokens 2048, seed 0; the original run used vLLM 0.10.x)
"""
import json, os, sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__))
M = sys.argv[1]; out = sys.argv[2]; think = sys.argv[3] == "think"
tok = AutoTokenizer.from_pretrained(M)
docs = [json.loads(l) for l in open(os.path.join(HERE, '..', 'data', 'ifeval.jsonl'))]
prompts = [tok.apply_chat_template([{"role": "user", "content": d["prompt"]}], tokenize=False,
           add_generation_prompt=True, enable_thinking=think) for d in docs]
llm = LLM(M, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=8192, seed=0)
sp = SamplingParams(temperature=0, max_tokens=6000 if think else 2048)
res = llm.generate(prompts, sp)
with open(out, "w") as f:
    for d, p, r in zip(docs, prompts, res):
        f.write(json.dumps({"key": d["key"], "prompt": d["prompt"], "chat": p, "response": r.outputs[0].text,
                            "finish": r.outputs[0].finish_reason, "ntok": len(r.outputs[0].token_ids)}) + "\n")

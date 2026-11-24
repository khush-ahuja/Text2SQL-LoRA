"""Compare the base model and the LoRA model on held-out questions.

    python evaluate.py --adapter runs/qwen05b-lora/adapter --n-test 500

Writes runs/.../eval.json (metrics) and eval_samples.jsonl (every prediction, for error analysis).
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from text2sql.data import load_splits, messages
from text2sql.metrics import summarize
from text2sql.sqlexec import exact_match, execution_match, extract_sql


@torch.no_grad()
def generate(model, tok, rows, batch=32, max_new=128):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        prompts = [tok.apply_chat_template(messages(r["question"], r["context"]), add_generation_prompt=True,
                                           tokenize=False) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {min(i + batch, len(rows))}/{len(rows)}", end="\r")
    return outs


def score(rows, raw_outputs):
    samples = []
    for r, raw in zip(rows, raw_outputs, strict=True):
        pred = extract_sql(raw)
        ex = execution_match(pred, r["answer"], r["context"])
        samples.append({"question": r["question"], "context": r["context"], "gold": r["answer"], "pred": pred,
                        "exact": exact_match(pred, r["answer"]), **ex})
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()

    _, test = load_splits(n_test=a.n_test)
    rows = list(test)
    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, device_map="auto")

    results, all_samples = {}, {}
    print("base model...")
    all_samples["base"] = score(rows, generate(model, tok, rows, a.batch))

    from peft import PeftModel

    print("\nLoRA model...")
    model = PeftModel.from_pretrained(model, a.adapter)
    all_samples["lora"] = score(rows, generate(model, tok, rows, a.batch))

    for name, samples in all_samples.items():
        results[name] = summarize(samples)
        print(f"\n{name}: {results[name]}")

    out = Path(a.adapter).parent
    (out / "eval.json").write_text(json.dumps({"n_test": len(rows), "base_model": a.model, **results}, indent=2))
    with open(out / "eval_samples.jsonl", "w") as f:
        for name, samples in all_samples.items():
            for s in samples:
                f.write(json.dumps({"model": name, **s}) + "\n")


if __name__ == "__main__":
    main()

"""Ask the fine-tuned model a question about your own schema.

    python infer.py --adapter runs/qwen05b-lora/adapter \
        --schema "CREATE TABLE employees (name VARCHAR, dept VARCHAR, salary INTEGER)" \
        --question "Which department has the highest average salary?"
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from text2sql.data import messages
from text2sql.sqlexec import extract_sql


def load(model_name: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok


@torch.no_grad()
def to_sql(model, tok, question: str, schema: str) -> str:
    ids = tok.apply_chat_template(messages(question, schema), add_generation_prompt=True, return_tensors="pt")
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = ids.to(model.device)
    out = model.generate(ids, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
    return extract_sql(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", help="LoRA adapter folder; omit to use the base model")
    p.add_argument("--schema", required=True)
    p.add_argument("--question", required=True)
    a = p.parse_args()
    model, tok = load(a.model, a.adapter)
    print(to_sql(model, tok, a.question, a.schema))


if __name__ == "__main__":
    main()

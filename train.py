"""LoRA fine-tuning of a small instruct model for text-to-SQL.

    python train.py --model Qwen/Qwen2.5-0.5B-Instruct --n-train 20000 --epochs 1 --out runs/qwen05b-lora

Fits on a free Colab T4 (16 GB): the base model is frozen, only low-rank adapters on the attention and
MLP projections are trained (~1% of the parameters).
"""
import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from text2sql.data import collate, encode, load_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--out", default="runs/qwen05b-lora")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train, _ = load_splits(n_train=a.n_train)
    train = train.map(lambda r: encode(tok, r["question"], r["context"], r["answer"], a.max_len),
                      remove_columns=train.column_names)

    # fp32 master weights + fp16 autocast: works on T4 (no bf16) without fp16-gradient errors
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=30, logging_steps=25, save_strategy="no", report_to=[],
        fp16=torch.cuda.is_available(), remove_unused_columns=False)
    t0 = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=train,
                      data_collator=partial(collate, pad_id=pad_id))
    result = trainer.train()

    out = Path(a.out)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    losses = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
    (out / "train_summary.json").write_text(json.dumps({
        "base_model": a.model, "n_train": len(train), "epochs": a.epochs, "lr": a.lr, "lora_rank": a.rank,
        "lora_alpha": a.alpha, "trainable_params": trainable, "total_params": total,
        "train_loss": result.training_loss, "minutes": round((time.time() - t0) / 60, 1), "loss_curve": losses,
    }, indent=2))
    print(f"saved adapter to {out / 'adapter'}")


if __name__ == "__main__":
    main()

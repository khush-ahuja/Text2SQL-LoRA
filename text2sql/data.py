"""Dataset loading, prompt formatting and label masking."""
import hashlib

DATASET = "b-mc2/sql-create-context"  # ~78k (question, CREATE TABLE context, SQL) triples, CC-BY-4.0

SYSTEM = "You translate questions into SQLite queries. Reply with a single SQL query and nothing else."


def user_prompt(question: str, context: str) -> str:
    return f"Schema:\n{context}\n\nQuestion: {question}"


def messages(question: str, context: str, answer: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(question, context)}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def split_of(row: dict, test_pct: int = 5) -> str:
    """Deterministic split by hashing the question+schema, so train and test never share an example
    and the split is the same on every machine."""
    h = int(hashlib.md5((row["question"] + row["context"]).encode()).hexdigest(), 16) % 100
    return "test" if h < test_pct else "train"


def load_splits(n_train: int | None = None, n_test: int | None = None, seed: int = 42):
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train")
    ds = ds.map(lambda r: {"split": split_of(r)})
    train = ds.filter(lambda r: r["split"] == "train").shuffle(seed=seed)
    test = ds.filter(lambda r: r["split"] == "test").shuffle(seed=seed)
    if n_train:
        train = train.select(range(min(n_train, len(train))))
    if n_test:
        test = test.select(range(min(n_test, len(test))))
    return train, test


def _ids(tokenizer, msgs, **kw) -> list[int]:
    out = tokenizer.apply_chat_template(msgs, tokenize=True, **kw)
    if hasattr(out, "keys"):  # newer transformers return a BatchEncoding
        out = out["input_ids"]
    return list(out)


def encode(tokenizer, question: str, context: str, answer: str, max_len: int = 384) -> dict:
    """Tokenizes one chat example and masks everything except the assistant's answer (label -100),
    so the model is trained only to produce the SQL, not to repeat the prompt."""
    prompt_ids = _ids(tokenizer, messages(question, context), add_generation_prompt=True)
    full_ids = _ids(tokenizer, messages(question, context, answer))
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template is not prefix-stable; cannot mask the prompt safely")
    full_ids = full_ids[:max_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pads a batch; padded positions get attention 0 and label -100."""
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        pad = width - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
    return {k: torch.tensor(v) for k, v in out.items()}

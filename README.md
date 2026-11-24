# Text2SQL-LoRA

[![tests](https://github.com/khush-ahuja/Text2SQL-LoRA/actions/workflows/tests.yml/badge.svg)](https://github.com/khush-ahuja/Text2SQL-LoRA/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khush-ahuja/Text2SQL-LoRA/blob/main/colab_train.ipynb)

Fine-tuning a small open LLM (**Qwen2.5-0.5B-Instruct**) to turn questions into SQL with **LoRA**, on a free Colab T4, and measuring whether it actually got better.

```
Schema:   CREATE TABLE head (name VARCHAR, born_state VARCHAR, age INTEGER)
Question: How many heads of departments are older than 56?
SQL:      SELECT COUNT(*) FROM head WHERE age > 56
```

## Results

Training and evaluation run in [`colab_train.ipynb`](colab_train.ipynb) on a free T4. The table below is filled from `eval.json` after the run (500 held-out questions, base model vs the same model with the LoRA adapter).

| Model | Execution accuracy | Exact match | Valid SQL |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct (base) | _pending run_ | | |
| + LoRA (20k examples, 1 epoch) | _pending run_ | | |

## How it works

**Data.** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context): ~78k (question, `CREATE TABLE` schema, SQL) examples. The train/test split is made by hashing each question + schema, so it is deterministic and no example appears in both.

**Training** (`train.py`)
- LoRA adapters (rank 16, alpha 32) on all attention and MLP projections; the 0.5B base model stays frozen
- Loss only on the SQL answer: prompt tokens are masked with `-100`, so the model learns to write SQL, not to repeat the schema
- fp32 master weights with fp16 autocast and gradient checkpointing, which fits a T4 (no bf16 support) without fp16-gradient errors
- Prompts use the model's own chat template; a check refuses templates that would make the prompt mask wrong

**Evaluation** (`evaluate.py`, `text2sql/sqlexec.py`)

Exact string match undercounts correct SQL (`COUNT(*)` vs `count(name)`, swapped conditions, `'x'` vs `"x"`), so the main metric is **execution accuracy**:

1. The dataset has schemas but no data, so the evaluator builds small SQLite databases from each schema, seeded with the literals that appear in the gold query (so `WHERE state = "California"` matches some rows) plus random values.
2. The predicted and gold queries are run on **three differently seeded databases**. A prediction counts as correct only if the results match on all three (order-insensitive unless the gold query has `ORDER BY`). This makes lucky matches much less likely, the same idea as test-suite accuracy (Zhong et al., 2020).
3. Gold queries that don't run are skipped and reported, not counted against the model.

Every prediction is saved to `eval_samples.jsonl` for error analysis.

## Run it

**Colab (free T4, about 25 minutes):** click the badge above, then Runtime → Run all.

**Locally (GPU):**
```bash
pip install -r requirements.txt
python train.py --n-train 20000 --epochs 1 --out runs/qwen05b-lora
python evaluate.py --adapter runs/qwen05b-lora/adapter --n-test 500
python infer.py --adapter runs/qwen05b-lora/adapter \
  --schema "CREATE TABLE employees (name VARCHAR, dept VARCHAR, salary INTEGER)" \
  --question "Which department has the highest average salary?"
```

## Tests

```bash
pytest
```

18 offline tests (no GPU or downloads) cover the evaluator (equivalent queries pass, wrong filters fail, invalid SQL, broken gold queries, ordering, float noise, SQL extraction from chatty replies) and the training data path (prompt masking, truncation, padding, deterministic split). CI runs ruff and the tests on Python 3.10–3.12.

## Limitations

- Single-table-heavy dataset with simple queries; this is not Spider/BIRD-level SQL.
- Synthetic databases can't catch every wrong query, so execution accuracy here is an upper bound on true correctness (checking on three databases makes this much tighter than one).
- 500 test questions gives roughly ±4 points of uncertainty on each accuracy number.

## Project layout

```
text2sql/
  data.py      dataset split, chat prompts, prompt masking, padding
  sqlexec.py   synthetic databases, execution match, exact match, SQL extraction
  metrics.py   aggregate scores
train.py       LoRA fine-tuning (transformers + peft)
evaluate.py    base vs LoRA comparison
infer.py       ask the model about your own schema
colab_train.ipynb
tests/
```

## Tech

Python · PyTorch · Hugging Face Transformers · PEFT (LoRA) · Datasets · SQLite · pytest · GitHub Actions · Google Colab

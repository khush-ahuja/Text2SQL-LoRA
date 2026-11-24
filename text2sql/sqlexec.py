"""Execution-based scoring for text-to-SQL.

Exact string match undercounts correct answers (`count(*)` vs `COUNT(*)`, swapped AND terms, aliases),
so the main metric here is execution accuracy: run the predicted and the gold query on the same
database and compare results.

The dataset only gives CREATE TABLE statements, not data, so we build small synthetic databases.
Rows are filled with the literals that appear in the gold query (so its WHERE filters match some rows)
plus random values. A prediction counts as correct only if it returns the same result as the gold
query on *every* one of several differently seeded databases; this makes lucky matches on one
database much less likely (the same idea as "test-suite accuracy" from Zhong et al., 2020).
"""
import random
import re
import sqlite3
from collections import Counter

_CREATE = re.compile(r"CREATE TABLE\s+(\w+)\s*\((.*?)\)", re.I | re.S)
_STR_LIT = re.compile(r"\"([^\"]*)\"|'([^']*)'")
_NUM_LIT = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def parse_schema(context: str) -> dict[str, list[tuple[str, str]]]:
    """'CREATE TABLE head (age INTEGER, name VARCHAR)' -> {'head': [('age', 'INTEGER'), ('name', 'VARCHAR')]}"""
    tables = {}
    for name, cols in _CREATE.findall(context):
        parsed = []
        for col in cols.split(","):
            parts = col.strip().split()
            if parts:
                parsed.append((parts[0], parts[1].upper() if len(parts) > 1 else "VARCHAR"))
        tables[name] = parsed
    return tables


def literals(sql: str) -> tuple[list[str], list[float]]:
    strings = [a or b for a, b in _STR_LIT.findall(sql)]
    without_strings = _STR_LIT.sub(" ", sql)
    numbers = [float(n) if "." in n else int(n) for n in _NUM_LIT.findall(without_strings)]
    return strings, numbers


def _value(ctype: str, strings: list[str], numbers: list, rng: random.Random):
    numeric = any(t in ctype for t in ("INT", "REAL", "FLOAT", "DOUBLE", "NUM", "DEC"))
    if numeric:
        if numbers and rng.random() < 0.5:
            return rng.choice(numbers)
        return rng.randint(-5, 100)
    if strings and rng.random() < 0.5:
        return rng.choice(strings)
    if numbers and rng.random() < 0.15:  # VARCHAR columns often hold numbers in this dataset
        return str(rng.choice(numbers))
    return rng.choice(["alpha", "beta", "gamma", "delta", "x", "y", "1", "2", "0"]) + str(rng.randint(0, 3))


def build_db(context: str, gold_sql: str, seed: int, rows: int = 12) -> sqlite3.Connection:
    rng = random.Random(seed)
    strings, numbers = literals(gold_sql)
    con = sqlite3.connect(":memory:")
    for table, cols in parse_schema(context).items():
        con.execute(f"CREATE TABLE {table} ({', '.join(f'{c} {t}' for c, t in cols)})")
        for _ in range(rows):
            vals = [_value(t, strings, numbers, rng) for _, t in cols]
            con.execute(f"INSERT INTO {table} VALUES ({', '.join('?' * len(vals))})", vals)
    return con


def _run(con: sqlite3.Connection, sql: str):
    try:
        return con.execute(sql).fetchall(), None
    except Exception as e:  # invalid SQL, unknown column, etc.
        return None, str(e)


def _norm_row(row):
    return tuple(round(v, 6) if isinstance(v, float) else v for v in row)


def same_result(a: list, b: list, ordered: bool) -> bool:
    a, b = [_norm_row(r) for r in a], [_norm_row(r) for r in b]
    return a == b if ordered else Counter(a) == Counter(b)


def normalize(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.replace("'", '"')
    return sql.lower()


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def execution_match(pred: str, gold: str, context: str, seeds=(0, 1, 2)) -> dict:
    """Returns {'correct': bool, 'valid': bool, 'error': str | None}."""
    ordered = "order by" in gold.lower()
    for seed in seeds:
        con = build_db(context, gold, seed)
        gold_rows, gold_err = _run(con, gold)
        pred_rows, pred_err = _run(con, pred)
        con.close()
        if gold_err:  # a few gold queries in public datasets don't run; skip them rather than blame the model
            return {"correct": None, "valid": pred_err is None, "error": "gold failed: " + gold_err}
        if pred_err:
            return {"correct": False, "valid": False, "error": pred_err}
        if not same_result(pred_rows, gold_rows, ordered):
            return {"correct": False, "valid": True, "error": None}
    return {"correct": True, "valid": True, "error": None}


def extract_sql(text: str) -> str:
    """Pulls the SQL out of a model reply (handles ```sql fences and chatty prefixes)."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    m = re.search(r"\b(SELECT|WITH)\b.*", text, re.S | re.I)
    if m:
        text = m.group(0)
    return text.strip().split(";")[0].strip()

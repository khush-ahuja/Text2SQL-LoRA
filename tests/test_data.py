import pytest

from text2sql.data import SYSTEM, collate, encode, messages, split_of
from text2sql.metrics import summarize


class FakeTokenizer:
    """Character-level stand-in for a chat tokenizer (prefix-stable like real chat templates)."""

    def _render(self, msgs, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</>" for m in msgs)
        return text + ("<assistant>" if add_generation_prompt else "")

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
        text = self._render(msgs, add_generation_prompt)
        return [ord(c) for c in text] if tokenize else text


def test_messages_have_system_schema_and_question():
    m = messages("How many heads?", "CREATE TABLE head (age INTEGER)", "SELECT COUNT(*) FROM head")
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert m[0]["content"] == SYSTEM and "CREATE TABLE head" in m[1]["content"]


def test_only_the_answer_is_trained_on():
    tok = FakeTokenizer()
    ex = encode(tok, "q?", "CREATE TABLE t (a INTEGER)", "SELECT a FROM t")
    learned = "".join(chr(i) for i, lab in zip(ex["input_ids"], ex["labels"], strict=True) if lab != -100)
    assert learned == "SELECT a FROM t</>"
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_truncation_keeps_lengths_aligned():
    ex = encode(FakeTokenizer(), "q?", "CREATE TABLE t (a INTEGER)", "SELECT a FROM t", max_len=20)
    assert len(ex["input_ids"]) == len(ex["labels"]) == 20


def test_non_prefix_stable_template_is_rejected():
    class Weird(FakeTokenizer):
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
            ids = super().apply_chat_template(msgs, tokenize, add_generation_prompt)
            return ids if add_generation_prompt else [0] + ids

    with pytest.raises(ValueError):
        encode(Weird(), "q", "CREATE TABLE t (a INTEGER)", "SELECT a FROM t")


def test_collate_pads_with_masked_labels():
    torch = pytest.importorskip("torch")
    batch = collate([{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
                     {"input_ids": [8], "attention_mask": [1], "labels": [8]}], pad_id=0)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert batch["input_ids"].dtype == torch.long


def test_split_is_deterministic_and_roughly_5_percent():
    rows = [{"question": f"q{i}", "context": "c"} for i in range(4000)]
    splits = [split_of(r) for r in rows]
    assert splits == [split_of(r) for r in rows]
    assert 0.03 < splits.count("test") / len(rows) < 0.07


def test_summarize_skips_bad_gold():
    s = [{"correct": True, "exact": True, "valid": True}, {"correct": False, "exact": False, "valid": False},
         {"correct": None, "exact": False, "valid": True}]
    assert summarize(s) == {"n_scored": 2, "execution_accuracy": 0.5, "exact_match": 0.5, "valid_sql": 0.5,
                            "skipped_bad_gold": 1}

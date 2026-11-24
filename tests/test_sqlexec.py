from text2sql.sqlexec import (
    build_db,
    exact_match,
    execution_match,
    extract_sql,
    literals,
    parse_schema,
    same_result,
)

CTX = "CREATE TABLE head (name VARCHAR, born_state VARCHAR, age INTEGER)"


def test_parse_schema_multiple_tables():
    ctx = "CREATE TABLE a (x INTEGER, y VARCHAR), CREATE TABLE b (z REAL)"
    assert parse_schema(ctx) == {"a": [("x", "INTEGER"), ("y", "VARCHAR")], "b": [("z", "REAL")]}


def test_literals_are_extracted_for_seeding():
    strings, numbers = literals('SELECT name FROM head WHERE born_state = "California" AND age > 56')
    assert strings == ["California"] and numbers == [56]


def test_synthetic_db_contains_gold_literals():
    con = build_db(CTX, 'SELECT name FROM head WHERE born_state = "California"', seed=0)
    assert con.execute("SELECT COUNT(*) FROM head WHERE born_state = 'California'").fetchone()[0] > 0


def test_equivalent_queries_are_correct_even_if_text_differs():
    gold = "SELECT COUNT(*) FROM head WHERE age > 56"
    pred = "select count(name) from head where 56 < age"
    assert not exact_match(pred, gold)
    assert execution_match(pred, gold, CTX)["correct"] is True


def test_wrong_filter_is_caught():
    gold = 'SELECT name FROM head WHERE born_state = "California"'
    pred = 'SELECT name FROM head WHERE born_state = "Texas"'
    assert execution_match(pred, gold, CTX) == {"correct": False, "valid": True, "error": None}


def test_invalid_sql_is_not_valid():
    res = execution_match("SELECT nme FROM head", "SELECT name FROM head", CTX)
    assert res["correct"] is False and res["valid"] is False and "nme" in res["error"]


def test_broken_gold_is_skipped_not_blamed_on_model():
    res = execution_match("SELECT name FROM head", "SELECT missing FROM head", CTX)
    assert res["correct"] is None and res["error"].startswith("gold failed")


def test_order_matters_only_with_order_by():
    assert same_result([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not same_result([(1,), (2,)], [(2,), (1,)], ordered=True)
    gold = "SELECT name FROM head ORDER BY age"
    assert execution_match("SELECT name FROM head ORDER BY age DESC", gold, CTX)["correct"] is False


def test_float_noise_is_ignored():
    assert same_result([(0.1 + 0.2,)], [(0.3,)], ordered=True)


def test_exact_match_normalization():
    assert exact_match("select  name from head where x = 'a';", 'SELECT name FROM head WHERE x = "a"')


def test_extract_sql_from_chatty_reply():
    assert extract_sql("Sure! Here it is:\n```sql\nSELECT name FROM head;\n```") == "SELECT name FROM head"
    assert extract_sql("The query is SELECT age FROM head; hope it helps") == "SELECT age FROM head"


def test_unbuildable_schema_is_skipped_not_crashing():
    res = execution_match("SELECT 1", "SELECT 1", "CREATE TABLE t (order VARCHAR)")
    assert res["correct"] is None and res["error"].startswith("schema failed")

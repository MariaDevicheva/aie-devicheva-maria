from __future__ import annotations

import pandas as pd
import numpy as np

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    assert not corr.empty or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# ================
# НОВЫЕ ТЕСТЫ ДЛЯ ЭВРИСТИК
# ================

def test_constant_column_detection():
    """Тест: обнаружение константной колонки"""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "status": ["active", "active", "active", "active"],  # константа
        "score": [10, 20, 30, 40]
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_constant_columns"] is True
    assert "status" in flags["constant_columns"]
    assert flags["n_constant_columns"] == 1


def test_high_cardinality_categorical_detection():
    """Тест: обнаружение колонки с высокой кардинальностью"""
    # 100 уникальных строк из 100 → unique_ratio = 100%
    df = pd.DataFrame({
        "user_id": range(100),
        "email": [f"user{i}@test.com" for i in range(100)]
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_high_cardinality_categoricals"] is True
    assert len(flags["high_cardinality_columns"]) == 1
    assert flags["high_cardinality_columns"][0]["name"] == "email"
    assert flags["high_cardinality_columns"][0]["unique_ratio"] == 1.0


def test_suspicious_id_duplicates_detection():
    """Тест: обнаружение дубликатов в ID-колонке"""
    df = pd.DataFrame({
        "order_id": [101, 102, 102, 103, 104],  # дубликат 102
        "amount": [100, 200, 200, 300, 400]
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["suspicious_id_duplicates"] is True
    assert "order_id" in flags["potential_id_columns"]


def test_many_zero_values_detection():
    """Тест: обнаружение колонки с множеством нулей (≥80% согласно вашей логике)"""
    # 9 нулей из 10 строк → 90% → должно сработать
    df = pd.DataFrame({
        "transaction_id": range(1, 11),
        "refund_amount": [0] * 9 + [500]  # 90% нулей
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_many_zero_values"] is True
    assert "refund_amount" in flags["many_zero_columns"]
    assert flags["n_many_zero_cols"] == 1


def test_no_false_positive_for_zeros():
    """Тест: 30% нулей НЕ должны вызывать флаг (потому что порог 80%)"""
    df = pd.DataFrame({
        "x": [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]  # 30% нулей
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    # При пороге >0.8 — флаг должен быть False
    assert flags["has_many_zero_values"] is False
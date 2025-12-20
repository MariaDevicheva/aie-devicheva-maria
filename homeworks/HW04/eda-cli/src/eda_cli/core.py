from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes

import json
from pathlib import Path


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
    min_missing_share: float = 0.3,
) -> Dict[str, Any]:
    """
    Эвристики «качества» данных:
    - слишком много пропусков;
    - подозрительно мало строк;
    - константные колонки;
    - колонки с очень высокой долей пропусков;
    - высокая кардинальность категориальных признаков.
    """
    flags: Dict[str, Any] = {}

    # Базовые эвристики
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # Эвристика 1: Проверка константных колонок
    constant_cols = []
    for col in summary.columns:
        if col.unique == 1 and col.non_null > 0:
            constant_cols.append(col.name)

    flags["has_constant_columns"] = len(constant_cols) > 0
    flags["constant_columns"] = constant_cols
    flags["n_constant_columns"] = len(constant_cols)

    # Эвристика 2: Проверка колонок с пропусками
    very_high_missing_cols = []
    high_missing_cols = []
    for col in summary.columns:
        if col.missing_share > 0.9:
            very_high_missing_cols.append(col.name)
        if col.missing_share > min_missing_share:  # ← ИСПОЛЬЗУЕМ ПАРАМЕТР CLI
            high_missing_cols.append(col.name)

    flags["has_very_high_missing_cols"] = len(very_high_missing_cols) > 0
    flags["very_high_missing_columns"] = very_high_missing_cols
    flags["n_very_high_missing_cols"] = len(very_high_missing_cols)

    flags["has_high_missing_cols"] = len(high_missing_cols) > 0
    flags["high_missing_columns"] = high_missing_cols
    flags["n_high_missing_cols"] = len(high_missing_cols)

    # Эвристика 3: Проверка высокой кардинальности категориальных признаков
    high_card_cols = []
    for col in summary.columns:
        is_categorical = col.dtype in ['object', 'category', 'string']
        if is_categorical and col.non_null > 0:
            unique_ratio = col.unique / summary.n_rows if summary.n_rows > 0 else 0
            if col.unique > 50 or unique_ratio > 0.8:
                high_card_cols.append({
                    'name': col.name,
                    'unique': col.unique,
                    'unique_ratio': round(unique_ratio, 3)
                })

    flags["has_high_cardinality_categoricals"] = len(high_card_cols) > 0
    flags["high_cardinality_columns"] = high_card_cols
    flags["n_high_cardinality_cols"] = len(high_card_cols)

    # Эвристика 4: Проверка на дубликаты ID
    suspicious_id_duplicates = False
    potential_id_columns = []

    for col in summary.columns:
        col_lower = col.name.lower()
        id_keywords = ['id', 'key', 'index', 'pk', 'identifier']
        is_potential_id = any(keyword in col_lower for keyword in id_keywords)
        if is_potential_id:
            potential_id_columns.append(col.name)
            if df is not None and col.non_null > 0:
                actual_unique = df[col.name].dropna().nunique()
                actual_non_null = df[col.name].notna().sum()
                if actual_unique < actual_non_null:
                    suspicious_id_duplicates = True
                    break

    flags["suspicious_id_duplicates"] = suspicious_id_duplicates
    flags["potential_id_columns"] = potential_id_columns

    # Эвристика 5: Проверка нулевых значений в числовых колонках
    many_zeros_cols = []
    if df is not None:
        for col in summary.columns:
            if col.is_numeric and col.non_null > 0:
                try:
                    zero_count = int((df[col.name] == 0).sum())
                    zero_ratio = zero_count / col.non_null if col.non_null > 0 else 0
                    if zero_ratio > 0.8:
                        many_zeros_cols.append(col.name)
                except (KeyError, TypeError):
                    continue
    else:
        for col in summary.columns:
            if col.is_numeric and col.non_null > 0:
                if col.min == 0 and col.max == 0:
                    many_zeros_cols.append(col.name)

    flags["has_many_zero_values"] = len(many_zeros_cols) > 0
    flags["many_zero_columns"] = many_zeros_cols
    flags["n_many_zero_cols"] = len(many_zeros_cols)

    # Расчёт quality_score с учётом новых эвристик
    score = 1.0
    score -= max_missing_share

    if summary.n_rows < 100:
        score -= 0.2
    if summary.n_cols > 100:
        score -= 0.1
    if flags["has_constant_columns"]:
        score -= 0.1 * min(flags["n_constant_columns"], 3)
    if flags["has_very_high_missing_cols"]:
        score -= 0.15 * min(flags["n_very_high_missing_cols"], 3)
    if flags["has_high_missing_cols"]:
        score -= 0.05 * min(flags["n_high_missing_cols"], 5)
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.05 * min(flags["n_high_cardinality_cols"], 4)
    if flags["suspicious_id_duplicates"]:
        score -= 0.3
    if flags["has_many_zero_values"]:
        score -= 0.05 * min(flags["n_many_zero_cols"], 4)

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = round(score, 3)

    if score >= 0.8:
        quality_category = "Отличное"
    elif score >= 0.6:
        quality_category = "Хорошее"
    elif score >= 0.4:
        quality_category = "Среднее"
    elif score >= 0.2:
        quality_category = "Плохое"
    else:
        quality_category = "Очень плохое"

    flags["quality_category"] = quality_category

    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)


def get_quality_report(flags: Dict[str, Any]) -> str:
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("АНАЛИЗ КАЧЕСТВА ДАННЫХ")
    report_lines.append("=" * 60)
    report_lines.append(f"ОБЩАЯ ОЦЕНКА: {flags['quality_score']} ({flags.get('quality_category', 'Неизвестно')})")
    report_lines.append("")
    report_lines.append("БАЗОВЫЕ МЕТРИКИ:")
    report_lines.append(f"  • Слишком мало строк (<100): {flags['too_few_rows']}")
    report_lines.append(f"  • Слишком много колонок (>100): {flags['too_many_columns']}")
    report_lines.append(f"  • Максимальная доля пропусков: {flags['max_missing_share']:.1%}")
    report_lines.append(f"  • Есть колонки с >50% пропусков: {flags['too_many_missing']}")
    report_lines.append("")
    report_lines.append("НОВЫЕ ЭВРИСТИКИ КАЧЕСТВА:")

    if flags["has_constant_columns"]:
        report_lines.append(f"Найдены константные колонки ({flags['n_constant_columns']}):")
        for col in flags["constant_columns"][:5]:
            report_lines.append(f"     - {col}")
        if len(flags["constant_columns"]) > 5:
            report_lines.append(f"     ... и еще {len(flags['constant_columns']) - 5} колонок")
    else:
        report_lines.append("Константных колонок не обнаружено")

    if flags["has_very_high_missing_cols"]:
        report_lines.append(f"Колонки с >90% пропусков ({flags['n_very_high_missing_cols']}):")
        for col in flags["very_high_missing_columns"][:3]:
            report_lines.append(f"     - {col}")
    else:
        report_lines.append("Колонок с >90% пропусков не обнаружено")

    if flags["has_high_missing_cols"]:
        report_lines.append(f"Колонки с пропусками выше порога ({flags['n_high_missing_cols']}):")
        for col in flags["high_missing_columns"][:3]:
            report_lines.append(f"     - {col}")

    if flags["has_high_cardinality_categoricals"]:
        report_lines.append(f"Категориальные колонки с высокой кардинальностью ({flags['n_high_cardinality_cols']}):")
        for col_info in flags["high_cardinality_columns"][:3]:
            report_lines.append(f"     - {col_info['name']}: {col_info['unique']} уникальных ({col_info['unique_ratio']:.1%})")
    else:
        report_lines.append("Проблем с кардинальностью категориальных признаков нет")

    if flags["suspicious_id_duplicates"]:
        report_lines.append("Возможные дубликаты в ID-колонке")
        if flags.get("potential_id_columns"):
            report_lines.append(f"  Проверьте колонки: {', '.join(flags['potential_id_columns'][:3])}")
    else:
        report_lines.append("Проблем с уникальностью ID не обнаружено")

    if flags["has_many_zero_values"]:
        report_lines.append(f"Числовые колонки с нулевыми значениями ({flags['n_many_zero_cols']}):")
        for col in flags["many_zero_columns"][:3]:
            report_lines.append(f"     - {col}")

    report_lines.append("")
    report_lines.append("=" * 60)
    return "\n".join(report_lines)


def generate_json_summary(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    quality_flags: Dict[str, Any],
    top_categories: Dict[str, pd.DataFrame],
    corr_matrix: pd.DataFrame,
    params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    problematic_cols = []
    problematic_cols.extend(quality_flags.get("constant_columns", []))
    problematic_cols.extend(quality_flags.get("very_high_missing_columns", []))
    min_missing_share = params.get("min_missing_share", 0.3) if params else 0.3
    if not missing_df.empty:
        high_missing = missing_df[missing_df["missing_share"] > min_missing_share]
        problematic_cols.extend(high_missing.index.tolist())
    high_card_cols = [col["name"] for col in quality_flags.get("high_cardinality_columns", [])]
    problematic_cols.extend(high_card_cols)
    problematic_cols.extend(quality_flags.get("many_zero_columns", []))
    problematic_cols = sorted(list(set(problematic_cols)))

    json_data = {
        "dataset_info": {
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "columns": [col.name for col in summary.columns],
            "column_types": {col.name: col.dtype for col in summary.columns}
        },
        "quality_assessment": {
            "quality_score": quality_flags.get("quality_score", 0.0),
            "quality_category": quality_flags.get("quality_category", "Неизвестно"),
            "flags": {
                "too_few_rows": quality_flags.get("too_few_rows", False),
                "too_many_columns": quality_flags.get("too_many_columns", False),
                "too_many_missing": quality_flags.get("too_many_missing", False),
                "has_constant_columns": quality_flags.get("has_constant_columns", False),
                "has_very_high_missing_cols": quality_flags.get("has_very_high_missing_cols", False),
                "has_high_cardinality_categoricals": quality_flags.get("has_high_cardinality_categoricals", False),
                "suspicious_id_duplicates": quality_flags.get("suspicious_id_duplicates", False),
                "has_many_zero_values": quality_flags.get("has_many_zero_values", False)
            }
        },
        "problematic_columns": {
            "all": problematic_cols,
            "by_type": {
                "constant_columns": quality_flags.get("constant_columns", []),
                "very_high_missing_columns": quality_flags.get("very_high_missing_columns", []),
                "high_missing_columns": missing_df[missing_df["missing_share"] > min_missing_share].index.tolist() if not missing_df.empty else [],
                "high_cardinality_columns": high_card_cols,
                "many_zero_columns": quality_flags.get("many_zero_columns", [])
            }
        },
        "generation_info": {
            "timestamp": pd.Timestamp.now().isoformat(),
            "cli_parameters": params or {}
        }
    }

    if not missing_df.empty and not missing_df[missing_df["missing_count"] > 0].empty:
        top_missing = missing_df.head(5).to_dict(orient="records")
        json_data["missing_values_summary"] = {
            "total_missing_cells": int(missing_df["missing_count"].sum()),
            "columns_with_missing": missing_df[missing_df["missing_count"] > 0].index.tolist(),
            "top_missing_columns": top_missing
        }

    if top_categories:
        json_data["categorical_summary"] = {
            "total_categorical_columns": len(top_categories),
            "columns": list(top_categories.keys())
        }

    if not corr_matrix.empty:
        json_data["correlation_summary"] = {
            "has_correlations": True,
            "matrix_shape": list(corr_matrix.shape)
        }

    return json_data


def save_json_summary(json_data: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    return output_path
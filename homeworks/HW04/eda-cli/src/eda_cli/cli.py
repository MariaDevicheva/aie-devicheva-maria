from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    get_quality_report,
    generate_json_summary,
    save_json_summary,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(10, help="Количество top-значений для категориальных признаков."),
    title: str = typer.Option("Отчет EDA", help="Заголовок отчёта."),
    min_missing_share: float = typer.Option(
        0.3,
        min=0.0,
        max=1.0,
        help="Порог доли пропусков для выделения проблемных колонок."
    ),
    quality_threshold: float = typer.Option(
        0.6,
        min=0.0,
        max=1.0,
        help="Порог оценки качества для рекомендаций."
    ),
    json_summary: bool = typer.Option(False, help="Сохранить компактную JSON-сводку по датасету."),
) -> None:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, max_columns=5, top_k=top_k_categories)

    # 🔥 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: передаём df и min_missing_share
    quality_flags = compute_quality_flags(
        summary,
        missing_df,
        df=df,  # ← теперь передаём данные
        min_missing_share=min_missing_share  # ← и порог
    )

    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    if json_summary:
        cli_params = {
            "max_hist_columns": max_hist_columns,
            "top_k_categories": top_k_categories,
            "title": title,
            "min_missing_share": min_missing_share,
            "quality_threshold": quality_threshold,
            "json_summary": json_summary,
            "sep": sep,
            "encoding": encoding
        }
        json_data = generate_json_summary(
            summary=summary,
            missing_df=missing_df,
            quality_flags=quality_flags,
            top_categories=top_cats,
            corr_matrix=corr_df,
            params=cli_params
        )
        json_path = save_json_summary(json_data, out_root / "summary.json")
        typer.echo(f"JSON-сводка сохранена: {json_path}")

    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n")
        f.write(f"Сгенерировано: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Параметры отчета\n")
        f.write(f"- Максимум гистограмм: `{max_hist_columns}`\n")
        f.write(f"- Top категорий: `{top_k_categories}`\n")
        f.write(f"- Порог пропусков: `{min_missing_share:.0%}`\n")
        f.write(f"- Порог качества: `{quality_threshold:.0%}`\n")
        f.write(f"- JSON-сводка: `{'Да' if json_summary else 'Нет'}`\n\n")

        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных\n\n")
        quality_report = get_quality_report(quality_flags)
        f.write(quality_report)
        f.write("\n\n")

        f.write(f"### Колонки с пропусками >{min_missing_share:.0%}\n\n")
        problematic_cols = missing_df[missing_df["missing_share"] > min_missing_share]
        if not problematic_cols.empty:
            f.write("| Колонка | Пропусков | Доля |\n")
            f.write("|---------|-----------|------|\n")
            for idx, row in problematic_cols.iterrows():
                f.write(f"| {idx} | {int(row['missing_count'])} | {row['missing_share']:.1%} |\n")
        else:
            f.write(f"Нет колонок с пропусками более {min_missing_share:.0%}.\n")
        f.write("\n")

        f.write("### Детальный анализ проблем\n\n")

        if quality_flags.get("has_constant_columns", False):
            f.write("**Константные колонки:**\n")
            for col in quality_flags.get("constant_columns", []):
                f.write(f"- `{col}` - только одно значение\n")
            f.write("\n")

        if quality_flags.get("has_high_cardinality_categoricals", False):
            f.write("**Колонки с высокой кардинальностью:**\n")
            for col_info in quality_flags.get("high_cardinality_columns", []):
                f.write(f"- `{col_info['name']}`: {col_info['unique']} уникальных значений ")
                f.write(f"({col_info['unique_ratio']:.1%} от всех строк)\n")
            f.write("\n")

        if quality_flags.get("suspicious_id_duplicates", False):
            f.write("**Внимание:** Возможные дубликаты в ID-колонке\n\n")

        if quality_flags.get("has_many_zero_values", False):
            f.write("**Колонки с нулевыми значениями:**\n")
            for col in quality_flags.get("many_zero_columns", []):
                f.write(f"- `{col}`\n")
            f.write("\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write(f"Всего колонок с пропусками: {len(missing_df[missing_df['missing_count'] > 0])}\n")
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(f"*Показывается top-{top_k_categories} значений для каждой категориальной колонки*\n\n")
            for col_name, cat_df in top_cats.items():
                f.write(f"### {col_name}\n")
                f.write("| Значение | Количество | Доля |\n")
                f.write("|----------|------------|------|\n")
                for _, row in cat_df.iterrows():
                    f.write(f"| {row['value']} | {int(row['count'])} | {row['share']:.1%} |\n")
                f.write(f"\n*Всего уникальных значений: {df[col_name].nunique()}*\n\n")
            f.write("Полные таблицы см. в папке `top_categories/`.\n\n")

        f.write("## Числовые признаки\n\n")
        numeric_cols = [col.name for col in summary.columns if col.is_numeric]
        if numeric_cols:
            f.write(f"**Всего числовых колонок:** {len(numeric_cols)}\n")
            f.write(f"**Гистограммы созданы для:** {min(max_hist_columns, len(numeric_cols))} колонок\n\n")
            f.write("### Основные статистики\n")
            f.write("| Колонка | Min | Max | Mean | Std | Пропуски |\n")
            f.write("|---------|-----|-----|------|-----|----------|\n")
            for col in summary.columns:
                if col.is_numeric:
                    missing_pct = f"{col.missing_share:.1%}" if col.missing > 0 else "0%"
                    f.write(f"| {col.name} | {col.min or '—'} | {col.max or '—'} | ")
                    f.write(f"{col.mean or '—'} | {col.std or '—'} | {missing_pct} |\n")
            f.write("\n")
            f.write("### Гистограммы\n")
            f.write("См. файлы `hist_*.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")
            strong_corrs = []
            for i in range(len(corr_df.columns)):
                for j in range(i + 1, len(corr_df.columns)):
                    corr = corr_df.iloc[i, j]
                    if abs(corr) > 0.7:
                        col1 = corr_df.columns[i]
                        col2 = corr_df.columns[j]
                        strong_corrs.append((col1, col2, corr))
            if strong_corrs:
                f.write("### Сильные корреляции (|r| > 0.7)\n")
                f.write("| Колонка 1 | Колонка 2 | Коэффициент |\n")
                f.write("|-----------|-----------|-------------|\n")
                for col1, col2, corr in strong_corrs[:5]:
                    f.write(f"| {col1} | {col2} | {corr:.3f} |\n")
                f.write("\n")

        f.write("## Заключение\n\n")
        f.write(f"**Общая оценка качества:** {quality_flags['quality_score']:.3f} ")
        f.write(f"({quality_flags['quality_category']})\n\n")

        if quality_flags['quality_score'] >= quality_threshold:
            f.write("**Качество данных удовлетворительное** для анализа.\n")
        else:
            f.write("**Качество данных требует внимания** перед анализом.\n")

        f.write("\n### Рекомендации\n")
        recommendations = []

        if quality_flags.get("has_constant_columns", False):
            recommendations.append("Удалить константные колонки")
        if quality_flags.get("has_very_high_missing_cols", False):
            recommendations.append("Проверить колонки с >90% пропусков")
        if problematic_cols.shape[0] > 0:
            recommendations.append(f"Обработать пропуски в {problematic_cols.shape[0]} колонках")
        if quality_flags.get("suspicious_id_duplicates", False):
            recommendations.append("Проверить уникальность ID")
        if quality_flags.get("has_high_cardinality_categoricals", False):
            recommendations.append("Рассмотреть преобразование колонок с высокой кардинальностью")

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        else:
            f.write("Данные готовы для дальнейшего анализа.\n")

    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"   Заголовок: '{title}'")
    typer.echo(f"   Настройки:")
    typer.echo(f"     - Гистограммы: {max_hist_columns} колонок")
    typer.echo(f"     - Top категорий: {top_k_categories}")
    typer.echo(f"     - Порог пропусков: {min_missing_share:.0%}")
    typer.echo(f"     - Порог качества: {quality_threshold:.0%}")
    typer.echo(f"     - JSON-сводка: {'Да' if json_summary else 'Нет'}")

    if json_summary:
        typer.echo(f"\nJSON-сводка содержит:")
        typer.echo(f"   - Размеры датасета: {summary.n_rows} строк, {summary.n_cols} колонок")
        typer.echo(f"   - Оценку качества: {quality_flags['quality_score']:.3f}")
        typer.echo(f"   - Список проблемных колонок")
        typer.echo(f"   - Параметры генерации отчёта")

    typer.echo(f"\nОсновные файлы:")
    typer.echo(f"   - {md_path}")
    if json_summary:
        typer.echo(f"   - {out_root / 'summary.json'}")
    typer.echo(f"   - Таблицы: summary.csv, missing.csv, correlation.csv")
    typer.echo(f"   - Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()
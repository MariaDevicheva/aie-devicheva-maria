from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_histograms_per_column(
    df: pd.DataFrame,
    out_dir: PathLike,
    max_columns: int = 6,
    bins: int = 20,
) -> List[Path]:
    """
    Для числовых колонок строит по отдельной гистограмме.
    Возвращает список путей к PNG.
    
    Args:
        df: DataFrame с данными
        out_dir: директория для сохранения
        max_columns: максимальное количество гистограмм для создания
        bins: количество бинов в гистограммах
    """
    out_dir = _ensure_dir(out_dir)
    numeric_df = df.select_dtypes(include="number")

    paths: List[Path] = []
    
    # Ограничиваем количество колонок согласно параметру max_columns
    columns_to_plot = list(numeric_df.columns[:max_columns])
    
    for i, name in enumerate(columns_to_plot):
        s = numeric_df[name].dropna()
        if s.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(s.values, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_title(f"Гистограмма: {name}", fontsize=14)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Частота", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Добавляем статистики
        stats_text = f"n={len(s):,}\nμ={s.mean():.2f}\nσ={s.std():.2f}"
        ax.text(0.95, 0.95, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)
        
        # Добавляем квантили
        q25, q50, q75 = s.quantile([0.25, 0.5, 0.75])
        ax.axvline(q25, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(q50, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(q75, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        
        # Добавляем легенду для квантилей
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', alpha=0.7, label='Распределение'),
            Patch(facecolor='red', alpha=0.3, label=f'Q1: {q25:.2f}'),
            Patch(facecolor='green', alpha=0.3, label=f'Медиана: {q50:.2f}'),
            Patch(facecolor='orange', alpha=0.3, label=f'Q3: {q75:.2f}')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        fig.tight_layout()
        out_path = out_dir / f"hist_{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        paths.append(out_path)
    
    # Если нет числовых колонок для построения
    if not paths:
        # Создаем информационный график
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 
               "Нет числовых колонок для построения гистограмм\n"
               f"или max_columns={max_columns} ограничивает вывод",
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, wrap=True)
        ax.set_title("Информация о гистограммах", fontsize=14)
        ax.axis('off')
        
        out_path = out_dir / "hist_no_numeric.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(out_path)

    return paths


def plot_missing_matrix(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Простая визуализация пропусков: где True=пропуск, False=значение.
    
    Args:
        df: DataFrame с данными
        out_path: путь для сохранения изображения
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Рисуем пустой график
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Пустой датасет", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        mask = df.isna().values
        fig, ax = plt.subplots(figsize=(min(14, df.shape[1] * 0.6), 6))
        
        # Создаем кастомную colormap
        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(['lightgreen', 'crimson'])
        
        im = ax.imshow(mask, aspect="auto", interpolation="none", cmap=cmap)
        ax.set_xlabel("Колонки", fontsize=12)
        ax.set_ylabel("Строки", fontsize=12)
        ax.set_title("Матрица пропусков", fontsize=14, pad=20)
        
        # Настройка подписей осей
        if df.shape[1] <= 20:
            ax.set_xticks(range(df.shape[1]))
            ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=9)
        else:
            # Для большого числа колонок показываем каждую 5-ю
            step = max(1, df.shape[1] // 20)
            ax.set_xticks(range(0, df.shape[1], step))
            ax.set_xticklabels(df.columns[::step], rotation=45, ha='right', fontsize=8)
        
        # Настройка оси Y для больших датасетов
        if df.shape[0] > 100:
            step_y = max(1, df.shape[0] // 10)
            ax.set_yticks(range(0, df.shape[0], step_y))
            ax.set_yticklabels(range(0, df.shape[0], step_y), fontsize=8)
        
        # Добавляем легенду
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='Данные присутствуют'),
            Patch(facecolor='crimson', label='Пропущенные данные')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Добавляем статистику пропусков
        total_missing = mask.sum()
        total_cells = mask.size
        missing_percentage = total_missing / total_cells * 100 if total_cells > 0 else 0
        
        stats_text = f"Всего пропусков: {total_missing:,} ({missing_percentage:.1f}%)"
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Тепловая карта корреляции числовых признаков.
    
    Args:
        df: DataFrame с данными
        out_path: путь для сохранения изображения
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 
               "Недостаточно числовых колонок для корреляции\n"
               f"Найдено: {numeric_df.shape[1]} числовых колонок\n"
               "Нужно минимум: 2",
               ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        corr = numeric_df.corr(numeric_only=True)
        
        # Определяем размер фигуры на основе количества колонок
        fig_size = min(14, corr.shape[1] * 0.8), min(10, corr.shape[0] * 0.8)
        fig, ax = plt.subplots(figsize=fig_size)
        

        
        # Маска для верхнего треугольника
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Создаем heatmap
        sns.heatmap(corr, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f', 
                   cmap='coolwarm',
                   center=0,
                   square=True, 
                   linewidths=0.5,
                   cbar_kws={'shrink': 0.8, 'label': 'Коэффициент корреляции'},
                   ax=ax)
        
        ax.set_title("Тепловая карта корреляций", fontsize=16, pad=20)
        
        # Настройка подписей
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        # Выделяем сильные корреляции
        strong_correlations = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.7:
                    strong_correlations.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        
        # Добавляем информацию о сильных корреляциях
        if strong_correlations:
            strong_text = "Сильные корреляции (|r| > 0.7):\n"
            for col1, col2, corr_val in strong_correlations[:3]:  # Показываем первые 3
                strong_text += f"{col1} - {col2}: {corr_val:.2f}\n"
            if len(strong_correlations) > 3:
                strong_text += f"... и еще {len(strong_correlations) - 3}"
            
            ax.text(0.02, -0.1, strong_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_top_categories_tables(
    top_cats: Dict[str, pd.DataFrame],
    out_dir: PathLike,
    max_display: int = 10,
) -> List[Path]:
    """
    Сохраняет top-k категорий по колонкам в отдельные CSV.
    
    Args:
        top_cats: словарь с таблицами категорий
        out_dir: директория для сохранения
        max_display: максимальное количество строк для сохранения в CSV
    """
    out_dir = _ensure_dir(out_dir)
    paths: List[Path] = []
    
    for name, table in top_cats.items():
        # Ограничиваем количество строк для сохранения
        limited_table = table.head(max_display)
        out_path = out_dir / f"top_values_{name}.csv"
        
        # Добавляем информацию о полном количестве уникальных значений
        if 'full_unique_count' not in table.columns:
            # Сохраняем таблицу
            limited_table.to_csv(out_path, index=False, encoding='utf-8')
        else:
            # Если есть дополнительная информация, сохраняем без нее
            limited_table.drop(columns=['full_unique_count'], errors='ignore').to_csv(
                out_path, index=False, encoding='utf-8'
            )
        
        # Также сохраняем текстовый файл с summary
        summary_path = out_dir / f"summary_{name}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Категориальная колонка: {name}\n")
            f.write(f"Всего уникальных значений: {len(table)}\n")
            f.write(f"Показывается top-{len(limited_table)} значений:\n\n")
            
            # Заголовок таблицы
            f.write(f"{'Значение':<30} {'Количество':<12} {'Доля':<10}\n")
            f.write("-" * 55 + "\n")
            
            # Данные
            for _, row in limited_table.iterrows():
                value = str(row['value'])[:30]  # Обрезаем длинные значения
                count = int(row['count'])
                share = float(row['share'])
                f.write(f"{value:<30} {count:<12} {share:<10.3f}\n")
        
        paths.append(out_path)
        paths.append(summary_path)
    
    return paths


def create_summary_visualization(
    df: pd.DataFrame,
    summary_stats: dict,
    out_dir: PathLike,
) -> List[Path]:
    """
    Создает сводную визуализацию с основными метриками датасета.
    
    Args:
        df: DataFrame с данными
        summary_stats: словарь со статистиками (из core.py)
        out_dir: директория для сохранения
    """
    out_dir = _ensure_dir(out_dir)
    paths: List[Path] = []
    
    # Создаем график с основными метриками
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 1. Распределение типов данных
    dtype_counts = df.dtypes.value_counts()
    axes[0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[0].set_title('Распределение типов данных', fontsize=12)
    
    # 2. Пропуски по колонкам (top-10)
    missing_counts = df.isna().sum().sort_values(ascending=False).head(10)
    if len(missing_counts) > 0:
        axes[1].barh(range(len(missing_counts)), missing_counts.values)
        axes[1].set_yticks(range(len(missing_counts)))
        axes[1].set_yticklabels(missing_counts.index)
        axes[1].set_xlabel('Количество пропусков')
        axes[1].set_title('Top-10 колонок по пропускам', fontsize=12)
    
    # 3. Качество данных (score gauge)
    quality_score = summary_stats.get('quality_score', 0)
    axes[2].axis('off')
    axes[2].text(0.5, 0.7, f"Оценка качества данных", 
                ha='center', va='center', fontsize=14, fontweight='bold')
    axes[2].text(0.5, 0.5, f"{quality_score:.3f}", 
                ha='center', va='center', fontsize=24, color='green' if quality_score > 0.6 else 'red')
    axes[2].text(0.5, 0.3, summary_stats.get('quality_category', 'Неизвестно'), 
                ha='center', va='center', fontsize=12)
    
    # 4. Размеры датасета
    axes[3].axis('off')
    info_text = f"Размер датасета:\n\n"
    info_text += f"Строк: {summary_stats.get('n_rows', 0):,}\n"
    info_text += f"Колонок: {summary_stats.get('n_cols', 0):,}\n"
    info_text += f"Память: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    axes[3].text(0.5, 0.5, info_text, 
                ha='center', va='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Сводная информация о датасете', fontsize=16)
    plt.tight_layout()
    
    out_path = out_dir / "dataset_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    paths.append(out_path)
    return paths

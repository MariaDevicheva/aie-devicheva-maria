# S04 – eda_cli: HTTP-сервис качества датасетов (FastAPI)

Расширенная версия проекта `eda-cli` из Семинара 03.

К существующему CLI-приложению для EDA добавлен **HTTP-сервис на FastAPI** с эндпоинтами `/health`, `/quality`, `/quality-from-csv`, а также **новыми эндпоинтами**, реализованными в рамках домашнего задания HW04:
- `/quality-flags-from-csv` — возвращает только флаги качества (включая расширенные эвристики из HW03);
- `/metrics` — агрегированная статистика по работе API.

Используется в рамках Семинара 04 курса «Инженерия ИИ».

---

## Связь с S03

Проект в S04 основан на том же пакете `eda_cli`, что и в S03:

- сохраняется структура `src/eda_cli/` и CLI-команда `eda-cli`;
- добавлен модуль `api.py` с FastAPI-приложением;
- в зависимости добавлены `fastapi`, `uvicorn[standard]` и `python-multipart`.

Цель S04 – показать, как поверх уже написанного EDA-ядра поднять простой HTTP-сервис с расширенными возможностями.

---

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему
- Браузер (для Swagger UI `/docs`) или любой HTTP-клиент:
  - `curl` / HTTP-клиент в IDE / Postman / Hoppscotch и т.п.

---

## Инициализация проекта

В корне проекта (каталог `homeworks/HW04/eda-cli`):

```bash
uv sync
## Инициализация проекта
 
В корне проекта (S03):
 
```bash
uv sync
```
 
Эта команда:
 
- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.
 
## Запуск CLI
 
### 1. **overview** - Краткий обзор датасета
 
```bash
uv run eda-cli overview data/example.csv
```
 
Параметры:
 
- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### 2. **report** - Полный EDA-отчёт
 
```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В каталоге `reports/`:
 
- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.
- `dataset_summary.png` – сводная инфографика датасета.

Новые параметры команды **report** 
| Параметр | Описание | По умолчанию | Влияние на отчёт |
|----------|----------|--------------|------------------|
| `--max-hist-columns` | Максимальное количество гистограмм для числовых колонок | `6` | Ограничивает число создаваемых гистограмм `hist_*.png` |
| `--top-k-categories` | Количество top значений для категориальных признаков | `10` | Определяет сколько значений показывать в таблицах категорий |
| `--title` | Заголовок отчёта | `"Отчет EDA"` | Используется как заголовок в `report.md` |
| `--min-missing-share` | Порог доли пропусков для проблемных колонок | `0.3` (30%) | Колонки с пропусками выше этого порога выделяются как проблемные |
| `--quality-threshold` | Порог оценки качества для рекомендаций | `0.6` (60%) | Определяет, какие рекомендации даются в заключении |

 

## Запуск HTTP-сервиса

HTTP-сервис реализован в модуле `eda_cli.api` на FastAPI.

### Запуск Uvicorn

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Пояснения:

- `eda_cli.api:app` - путь до объекта FastAPI `app` в модуле `eda_cli.api`;
- `--reload` - автоматический перезапуск сервера при изменении кода (удобно для разработки);
- `--port 8000` - порт сервиса (можно поменять при необходимости).

После запуска сервис будет доступен по адресу:

```text
http://127.0.0.1:8000
```
---

## Эндпоинты сервиса

### 1. `GET /health`

Простейший health-check.

**Запрос:**

```http
GET /health
```

**Ожидаемый ответ `200 OK` (JSON):**

```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

Пример проверки через `curl`:

```bash
curl http://127.0.0.1:8000/health
```

---

### 2. Swagger UI: `GET /docs`

Интерфейс документации и тестирования API:

```text
http://127.0.0.1:8000/docs
```

Через `/docs` можно:

- вызывать `GET /health`;
- вызывать `POST /quality` (форма для JSON);
- вызывать `POST /quality-from-csv` (форма для загрузки файла).

---

### 3. `POST /quality` – заглушка по агрегированным признакам

Эндпоинт принимает **агрегированные признаки датасета** (размеры, доля пропусков и т.п.) и возвращает эвристическую оценку качества.

**Пример запроса:**

```http
POST /quality
Content-Type: application/json
```

Тело:

```json
{
  "n_rows": 10000,
  "n_cols": 12,
  "max_missing_share": 0.15,
  "numeric_cols": 8,
  "categorical_cols": 4
}
```

**Пример ответа `200 OK`:**

```json
{
  "ok_for_model": true,
  "quality_score": 0.8,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 3.2,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 10000,
    "n_cols": 12
  }
}
```

**Пример вызова через `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/quality" \
  -H "Content-Type: application/json" \
  -d '{"n_rows": 10000, "n_cols": 12, "max_missing_share": 0.15, "numeric_cols": 8, "categorical_cols": 4}'
```

---

### 4. `POST /quality-from-csv` – оценка качества по CSV-файлу

Эндпоинт принимает CSV-файл, внутри:

- читает его в `pandas.DataFrame`;
- вызывает функции из `eda_cli.core`:

  - `summarize_dataset`,
  - `missing_table`,
  - `compute_quality_flags`;
- возвращает оценку качества датасета в том же формате, что `/quality`.

**Запрос:**

```http
POST /quality-from-csv
Content-Type: multipart/form-data
file: <CSV-файл>
```

Через Swagger:

- в `/docs` открыть `POST /quality-from-csv`,
- нажать `Try it out`,
- выбрать файл (например, `data/example.csv`),
- нажать `Execute`.

**Пример вызова через `curl` (Linux/macOS/WSL):**

```bash
curl -X POST "http://127.0.0.1:8000/quality-from-csv" \
  -F "file=@data/example.csv"
```

Ответ будет содержать:

- `ok_for_model` - результат по эвристикам;
- `quality_score` - интегральный скор качества;
- `flags` - булевы флаги из `compute_quality_flags`;
- `dataset_shape` - реальные размеры датасета (`n_rows`, `n_cols`);
- `latency_ms` - время обработки запроса.

### 5. `POST /quality-flags-from-csv` – только флаги качества (HW04)

Эндпоинт принимает CSV-файл и возвращает **исключительно булевы флаги качества данных**, включая расширенные эвристики, реализованные в HW03:
- `has_constant_columns` — наличие колонок без вариативности;
- `has_high_cardinality_categoricals` — категориальные признаки с чрезмерно высокой кардинальностью;
- `suspicious_id_duplicates` — возможные дубликаты в ID-подобных колонках;
- `has_many_zero_values` — числовые колонки, в которых большинство значений — нули;
- и другие базовые флаги (`too_few_rows`, `too_many_missing` и т.д.).

**Запрос:**

```http
POST /quality-flags-from-csv
Content-Type: multipart/form-data
file: <CSV-файл>
```

Через Swagger:

- в /docs открыть POST /quality-flags-from-csv,
- нажать Try it out,
- выбрать файл (например, data/example.csv),
- нажать Execute.

**Пример вызова через `curl` (Linux/macOS/WSL):**

```bash
curl -X POST "http://127.0.0.1:8000/quality-flags-from-csv" \
  -F "file=@data/example.csv"
```

Ответ будет содержать:

- **flags** — словарь с булевыми значениями всех применённых эвристик качества.

Пример ответа:
```json
{
  "flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": true,
    "suspicious_id_duplicates": false,
    "has_many_zero_values": false
  }
}
```

## Тесты

* Запуск всех тестов 
```bash
uv run pytest -q
```
* Запуск с подробным выводом
```bash
uv run pytest -v
```


from __future__ import annotations

from time import perf_counter
import io
import logging
import uuid
from datetime import datetime
import json
import os
from collections import defaultdict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset

# === Глобальные переменные для метрик (Вариант F) ===
metrics = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "last_ok_for_model": None,
    "endpoint_counts": defaultdict(int),
}

# === Настройка логгера (Вариант D) ===
logger = logging.getLogger("api_requests")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Вывод в stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    
    # Вывод в файл logs/api.log
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/api.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)


def log_and_track_request(
    endpoint: str,
    status: str,
    latency_ms: float,
    ok_for_model: bool | None = None,
    n_rows: int | None = None,
    n_cols: int | None = None,
):
    """Логирует запрос в JSON и обновляет глобальные метрики."""
    # Обновляем метрики
    metrics["total_requests"] += 1
    metrics["total_latency_ms"] += latency_ms
    metrics["endpoint_counts"][endpoint] += 1
    if ok_for_model is not None:
        metrics["last_ok_for_model"] = ok_for_model

    # Формируем JSON-лог
    log_entry = {
        "endpoint": endpoint,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": str(uuid.uuid4()),
    }
    if ok_for_model is not None:
        log_entry["ok_for_model"] = ok_for_model
    if n_rows is not None:
        log_entry["n_rows"] = n_rows
    if n_cols is not None:
        log_entry["n_cols"] = n_cols

    logger.info(json.dumps(log_entry, ensure_ascii=False))


app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)


# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ..., ge=0.0, le=1.0, description="Максимальная доля пропусков среди всех колонок (0..1)"
    )
    numeric_cols: int = Field(..., ge=0, description="Количество числовых колонок")
    categorical_cols: int = Field(..., ge=0, description="Количество категориальных колонок")


class QualityResponse(BaseModel):
    ok_for_model: bool = Field(
        ..., description="True, если датасет считается достаточно качественным для обучения модели"
    )
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Интегральная оценка качества данных (0..1)")
    message: str = Field(..., description="Человекочитаемое пояснение решения")
    latency_ms: float = Field(..., ge=0.0, description="Время обработки запроса на сервере, миллисекунды")
    flags: dict[str, bool] | None = Field(
        default=None, description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)"
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None, description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны"
    )


# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


# ---------- /quality ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    start = perf_counter()

    score = 1.0
    score -= req.max_missing_share
    if req.n_rows < 1000:
        score -= 0.2
    if req.n_cols > 100:
        score -= 0.1
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    message = (
        "Данных достаточно, модель можно обучать (по текущим эвристикам)."
        if ok_for_model
        else "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."
    )

    latency_ms = (perf_counter() - start) * 1000.0
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # Заменяем print → структурированное логирование
    log_and_track_request(
        endpoint="/quality",
        status="success",
        latency_ms=latency_ms,
        ok_for_model=ok_for_model,
        n_rows=req.n_rows,
        n_cols=req.n_cols,
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv ----------


@app.post("/quality-from-csv", response_model=QualityResponse, tags=["quality"])
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    message = (
        "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
        if ok_for_model
        else "CSV требует доработки перед обучением модели (по текущим эвристикам)."
    )

    latency_ms = (perf_counter() - start) * 1000.0
    flags_bool = {key: bool(value) for key, value in flags_all.items() if isinstance(value, bool)}

    n_rows = getattr(summary, "n_rows", df.shape[0])
    n_cols = getattr(summary, "n_cols", df.shape[1])

    # Заменяем print → структурированное логирование
    log_and_track_request(
        endpoint="/quality-from-csv",
        status="success",
        latency_ms=latency_ms,
        ok_for_model=ok_for_model,
        n_rows=n_rows,
        n_cols=n_cols,
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- /quality-flags-from-csv ----------


@app.post("/quality-flags-from-csv", tags=["quality"])
async def quality_flags_from_csv(file: UploadFile = File(...)) -> dict:
    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        return JSONResponse(status_code=400, content={"error": "Ожидается CSV-файл (content-type text/csv)."})

    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"error": "Пустой файл"})
        df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Не удалось прочитать CSV: {str(exc)}"})

    if df.empty:
        return JSONResponse(status_code=400, content={"error": "CSV-файл пуст"})

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df, df=df)

    boolean_flags = {key: value for key, value in flags_all.items() if isinstance(value, bool)}

    latency_ms = (perf_counter() - start) * 1000.0
    log_and_track_request(
        endpoint="/quality-flags-from-csv",
        status="success",
        latency_ms=latency_ms,
        n_rows=len(df),
        n_cols=len(df.columns),
    )

    return {"flags": boolean_flags}


# ---------- /metrics (Вариант F) ----------


@app.get("/metrics", tags=["system"], summary="Простая статистика по работе API")
def get_metrics():
    if metrics["total_requests"] == 0:
        avg_latency = 0.0
    else:
        avg_latency = metrics["total_latency_ms"] / metrics["total_requests"]

    return {
        "total_requests": metrics["total_requests"],
        "avg_latency_ms": round(avg_latency, 2),
        "last_ok_for_model": metrics["last_ok_for_model"],
        "requests_by_endpoint": dict(metrics["endpoint_counts"]),
    }
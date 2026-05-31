"""
PricePulse API - Сервис интеллектуального мониторинга цен
"""
import logging
import time
import os
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ========== Настройка логирования ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PricePulse")

# ========== FastAPI приложение ==========
app = FastAPI(
    title="PricePulse API",
    description="Сервис прогнозирования оптимальной розничной цены",
    version="1.0.0"
)

# ========== Prometheus метрики ==========
PREDICTION_COUNTER = Counter(
    'predictions_total',
    'Общее количество прогнозов',
    ['category']
)
PREDICTION_TIME = Histogram(
    'prediction_duration_seconds',
    'Время выполнения прогноза'
)
ERROR_COUNTER = Counter(
    'prediction_errors_total',
    'Общее количество ошибок'
)

# ========== Загрузка модели ==========
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'price_optimizer.pkl')

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Модель загружена из {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Модель не найдена по пути: {MODEL_PATH}")
    model = None

# ========== Pydantic модели ==========
class PriceInput(BaseModel):
    """Входные данные для прогноза"""
    competitor_price: float = Field(..., description="Цена конкурента (руб.)", ge=0)
    our_price: float = Field(..., description="Наша текущая цена (руб.)", ge=0)
    demand_index: int = Field(..., description="Индекс спроса (50-150)", ge=50, le=150)
    stock_level: int = Field(..., description="Уровень запасов", ge=0)
    margin: float = Field(..., description="Маржинальность (0.1-0.4)", ge=0.1, le=0.4)
    category: str = Field(..., description="Категория товара")

    model_config = {
        "json_schema_extra": {
            "example": {
                "competitor_price": 2500.0,
                "our_price": 2550.0,
                "demand_index": 110,
                "stock_level": 500,
                "margin": 0.25,
                "category": "электроника"
            }
        }
    }

class PriceOutput(BaseModel):
    """Результат прогноза"""
    optimal_price: float = Field(..., description="Рекомендованная оптимальная цена (руб.)")
    current_price: float = Field(..., description="Текущая цена")
    competitor_price: float = Field(..., description="Цена конкурента")
    price_difference: float = Field(..., description="Разница с оптимальной ценой")
    recommendation: str = Field(..., description="Рекомендация по изменению цены")
    price_position: str = Field(..., description="Позиция относительно рынка")
    timestamp: str = Field(..., description="Время прогноза")

class HealthResponse(BaseModel):
    """Ответ health-check"""
    status: str
    model_loaded: bool
    version: str
    timestamp: str

# ========== Endpoints ==========
@app.get("/", response_model=dict)
async def root():
    """Корневой endpoint"""
    return {
        "message": "Добро пожаловать в PricePulse API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка работоспособности сервиса"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PriceOutput)
async def predict_price(input_data: PriceInput):
    """
    Прогноз оптимальной цены товара.
    
    Принимает параметры товара и возвращает рекомендованную цену
    с указанием позиционирования относительно рынка.
    """
    if model is None:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    start_time = time.time()
    
    try:
        # Подготовка данных
        features = pd.DataFrame([{
            'competitor_price': input_data.competitor_price,
            'our_price': input_data.our_price,
            'demand_index': input_data.demand_index,
            'stock_level': input_data.stock_level,
            'margin': input_data.margin,
            'category': input_data.category
        }])
        
        # Прогноз
        prediction = model.predict(features)[0]
        optimal_price = round(float(prediction), 2)
        
        # Разница с текущей ценой
        price_diff = optimal_price - input_data.our_price
        diff_percent = (price_diff / input_data.our_price) * 100
        
        # Определяем позицию НАШЕЙ цены относительно конкурента
        if input_data.our_price < input_data.competitor_price * 0.95:
            position = "Ниже рынка"
        elif input_data.our_price > input_data.competitor_price * 1.05:
            position = "Выше рынка"
        else:
            position = "Рыночная"
        
        # Формируем рекомендацию на основе сравнения optimal_price и our_price
        if optimal_price > input_data.our_price:
            recommendation = f"Рекомендуется повысить цену до {optimal_price:.0f} руб. (+{abs(diff_percent):.1f}%)"
        elif optimal_price < input_data.our_price:
            recommendation = f"Рекомендуется снизить цену до {optimal_price:.0f} руб. ({diff_percent:.1f}%)"
        else:
            recommendation = f"Цена оптимальна: {optimal_price:.0f} руб."
        
        # Обновление метрик
        PREDICTION_COUNTER.labels(category=input_data.category).inc()
        PREDICTION_TIME.observe(time.time() - start_time)
        
        logger.info(
            f"Прогноз: категория={input_data.category}, "
            f"тек.цена={input_data.our_price:.0f}, "
            f"оптим.цена={optimal_price:.0f}, "
            f"позиция={position}"
        )
        
        return PriceOutput(
            optimal_price=optimal_price,
            current_price=input_data.our_price,
            competitor_price=input_data.competitor_price,
            price_difference=round(price_diff, 2),
            recommendation=recommendation,
            price_position=position,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Ошибка прогноза: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка прогнозирования: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Метрики Prometheus"""
    return JSONResponse(
        content=generate_latest().decode('utf-8'),
        media_type=CONTENT_TYPE_LATEST
    )

# ========== Запуск ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
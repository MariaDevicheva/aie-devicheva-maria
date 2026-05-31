"""
Тесты для PricePulse API
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    """Тест корневого endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    print("test_root пройден")


def test_health_check():
    """Тест health-check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    assert data["version"] == "1.0.0"
    print("test_health_check пройден")


def test_predict_success():
    """Тест успешного прогноза"""
    test_data = {
        "competitor_price": 2500.0,
        "our_price": 2550.0,
        "demand_index": 110,
        "stock_level": 500,
        "margin": 0.25,
        "category": "электроника"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "optimal_price" in data
    assert data["optimal_price"] > 0
    assert "recommendation" in data
    assert "price_position" in data
    assert "timestamp" in data
    
    print(f"   test_predict_success пройден")
    print(f"   optimal_price: {data['optimal_price']:.2f} руб.")
    print(f"   price_position: {data['price_position']}")
    print(f"   recommendation: {data['recommendation']}")


def test_predict_different_categories():
    """Тест прогноза для разных категорий"""
    categories = ["электроника", "бытовая техника", "аксессуары"]
    
    for cat in categories:
        test_data = {
            "competitor_price": 1000.0,
            "our_price": 1050.0,
            "demand_index": 100,
            "stock_level": 300,
            "margin": 0.2,
            "category": cat
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert data["optimal_price"] > 0
        
    print(f" test_predict_different_categories пройден ({len(categories)} категорий)")


def test_predict_invalid_input():
    """Тест с некорректными данными"""
    # Пустой запрос
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error
    print(" test_predict_invalid_input (пустой) пройден")
    
    # Отрицательная цена
    response = client.post("/predict", json={
        "competitor_price": -100.0,
        "our_price": 100.0,
        "demand_index": 100,
        "stock_level": 100,
        "margin": 0.2,
        "category": "электроника"
    })
    assert response.status_code == 422
    print(" test_predict_invalid_input (отриц. цена) пройден")


def test_metrics():
    """Тест метрик Prometheus"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text
    print(" test_metrics пройден")


# ========== Запуск всех тестов ==========
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("ЗАПУСК ТЕСТОВ PricePulse API")
    print("=" * 50 + "\n")
    
    tests = [
        test_root,
        test_health_check,
        test_predict_success,
        test_predict_different_categories,
        test_predict_invalid_input,
        test_metrics
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f" {test.__name__} ПРОВАЛЕН: {e}")
        except Exception as e:
            failed += 1
            print(f" {test.__name__} ОШИБКА: {e}")
    
    print("\n" + "=" * 50)
    print(f"РЕЗУЛЬТАТ: {passed} пройдено, {failed} провалено")
    print("=" * 50)
import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 20

def test_health():
    """Test health check endpoint"""
    print("\n=== Test 1: Health Check ===")
    response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ PASSED")


def test_root():
    """Test root endpoint"""
    print("\n=== Test 2: Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert "message" in response.json()
    print("✓ PASSED")


def test_predict_english_positive():
    """Test predict endpoint with English positive text"""
    print("\n=== Test 3: English Positive Text ===")
    payload = {"text": "I absolutely love this product! It's amazing!"}
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    print(f"Input: {payload['text']}")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    assert response.status_code == 200
    assert "label" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    print("✓ PASSED")


def test_predict_english_negative():
    """Test predict endpoint with English negative text"""
    print("\n=== Test 4: English Negative Text ===")
    payload = {"text": "I hate this! This is terrible and awful."}
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    print(f"Input: {payload['text']}")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    assert response.status_code == 200
    assert "label" in result
    assert "confidence" in result
    print("✓ PASSED")


def test_predict_vietnamese_positive():
    """Test predict endpoint with Vietnamese positive text"""
    print("\n=== Test 5: Vietnamese Positive Text ===")
    payload = {"text": "Tôi rất yêu thích sản phẩm này! Nó tuyệt vời!"}
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    print(f"Input: {payload['text']}")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    assert response.status_code == 200
    assert "label" in result
    assert "confidence" in result
    print("✓ PASSED")


def test_predict_vietnamese_negative():
    """Test predict endpoint with Vietnamese negative text"""
    print("\n=== Test 6: Vietnamese Negative Text ===")
    payload = {"text": "Tôi ghét cái sản phẩm này. Nó rất tệ!"}
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    print(f"Input: {payload['text']}")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    assert response.status_code == 200
    assert "label" in result
    assert "confidence" in result
    print("✓ PASSED")


def test_predict_neutral():
    """Test predict endpoint with neutral text"""
    print("\n=== Test 7: Neutral Text ===")
    payload = {"text": "The weather is cloudy today."}
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    print(f"Input: {payload['text']}")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    assert response.status_code == 200
    assert "label" in result
    assert "confidence" in result
    print("✓ PASSED")


def test_predict_empty_text():
    """Test predict endpoint with empty text (should fail)"""
    print("\n=== Test 8: Empty Text (Error Case) ===")
    payload = {"text": ""}
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    print(f"Input: {payload['text']}")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 400  # Should return Bad Request
    print("✓ PASSED (Error handled correctly)")


def test_predict_whitespace_only():
    """Test predict endpoint with whitespace only (should fail)"""
    print("\n=== Test 9: Whitespace Only (Error Case) ===")
    payload = {"text": "   \n\t  "}
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    print(f"Input: '{payload['text']}'")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 400
    print("✓ PASSED (Error handled correctly)")


def test_predict_long_text():
    """Test predict endpoint with longer text"""
    print("\n=== Test 10: Long Text ===")
    payload = {"text": "This is a wonderful and fantastic product! " * 5}
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    print(f"Input length: {len(payload['text'])} characters")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    assert response.status_code == 200
    assert "label" in result
    assert "confidence" in result
    print("✓ PASSED")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Sentiment Analysis API Test Suite")
    print("=" * 60)
    
    tests = [
        test_health,
        test_root,
        test_predict_english_positive,
        test_predict_english_negative,
        test_predict_vietnamese_positive,
        test_predict_vietnamese_negative,
        test_predict_neutral,
        test_predict_empty_text,
        test_predict_whitespace_only,
        test_predict_long_text,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except requests.exceptions.ConnectionError:
            print("✗ ERROR: Could not connect to API. Make sure it's running at http://localhost:8000")
            print("  Run: uvicorn src.main:app --reload")
            return
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

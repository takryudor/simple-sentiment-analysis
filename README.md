# simple-sentiment-analysis

## Student info

- **Fullname:** Duong Phan Minh Tri
- **Student ID:** 24120470
- **Email:** 24120470@student.hcmus.edu.vn
- **Course's class:** 24CTT3(B)

## Model Information

**Model:** `tabularisai/multilingual-sentiment-analysis`

- Trained on multiple languages
- Fine-tuned for sentiment classification
- Supports 5 sentiment classes (Very Negative → Very Positive)

**Link to Hugging Face:** [tabularisai/multilingual-sentiment-analysis](https://huggingface.co/tabularisai/multilingual-sentiment-analysis)

## Description

A multilingual sentiment analysis API built with **FastAPI** and **Hugging Face Transformers**. This application analyzes text sentiment in multiple languages (Vietnamese, English, etc.) and returns predictions with confidence scores.

**Supported sentiment labels:**

- `Very Negative`
- `Negative`
- `Neutral`
- `Positive`
- `Very Positive`

## Project Structure

```
simple-sentiment-analysis/
├── src/
│   ├── __init__.py
│   ├── main.py            # FastAPI app initialization & endpoints
│   └── services.py        # Hugging Face model loading & predictions
├── tests/
│   └── test_api.py        # API integration/smoke tests (requests)
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # <-- You are here
```

## Features

- 🌍 **Multilingual Support**: Works with English, Vietnamese, and other languages
- 🚀 **FastAPI Server**: Fast, modern Python web framework
- 🤖 **Pre-trained Model**: Uses `tabularisai/multilingual-sentiment-analysis` from Hugging Face
- 📊 **Confidence Scores**: Returns sentiment label with confidence percentage
- ✅ **Error Handling**: Validates input and returns meaningful error messages

## Setup & Installation

### Requirements

- Python 3.8+ (recommended: Python 3.10 or newer)
- pip (Python package manager)

### Step 1: Clone & Navigate

```bash
git clone https://github.com/takryudor/simple-sentiment-analysis
cd simple-sentiment-analysis
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the API Server

```bash
uvicorn src.main:app --reload
```

The API will start at: `http://localhost:8000`

**Swagger UI** (Interactive API docs): `http://localhost:8000/docs`

## API Endpoints

### 1. Root Endpoint

```
GET /
```

**Response:**

```json
{
  "message": "Chào mừng bạn đến với API phân tích cảm xúc đa ngôn ngữ sử dụng Hugging Face, nơi bạn có thể kiểm tra trạng thái hệ thống và dự đoán cảm xúc của văn bản chỉ với vài bước đơn giản."
}
```

### 2. Health Check

```
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Predict Sentiment

```
POST /predict
```

**Request Body:**

```json
{
  "text": "I love this product!"
}
```

**Success Response (200):**

```json
{
  "input": "I love this product!",
  "label": "Positive",
  "confidence": 0.9876
}
```

**Error Response (400 - Empty text):**

```json
{
  "detail": "Văn bản không được để trống"
}
```

**Error Response (422 - Validation error, e.g. non-string input):**

```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "Input should be a valid string",
      "type": "string_type"
    }
  ]
}
```

**Error Response (500 - Internal prediction error):**

```json
{
  "detail": "Prediction failed"
}
```

## Usage Examples

### Using cURL

```bash
# Test health check
curl http://localhost:8000/health

# Test sentiment prediction (English)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'

# Test sentiment prediction (Vietnamese)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Tôi rất yêu thích sản phẩm này!"}'
```

### Using Python + Requests

```python
import requests

url = "http://localhost:8000/predict"
payload = {"text": "I hate this terrible experience"}

response = requests.post(url, json=payload)
print(response.json())
# Output: {"input": "...", "label": "Negative", "confidence": 0.8765}
```

### Using Swagger UI

1. Open `http://localhost:8000/docs` in your browser
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Enter text in the request body
5. Click "Execute"

## Testing

Please make sure you ran

```
uvicorn src.main:app --reload
```

Then run

```bash
python tests/test_api.py
```

## Demo examples

### Vietnamese Examples

- **Input**: "Tôi rất yêu thích sản phẩm này!" → Expected: **Positive** ✅
- **Input**: "Tôi ghét cái sản phẩm này" → Expected: **Negative** ✅
- **Input**: "Thời tiết hôm nay là thời tiết đẹp" → Expected: **Neutral** ✅

### English Examples

- **Input**: "This product is absolutely amazing!" → Expected: **Very Positive** ✅
- **Input**: "I hate this terrible experience" → Expected: **Negative** ✅
- **Input**: "The sky is blue" → Expected: **Neutral** ✅

## Video demo

-> **Youtube video**: [simple-sentiment-analysis](https://youtube.com)

## Troubleshooting

| Issue                                            | Solution                                                                                   |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError: No module named 'fastapi'` | Run `pip install -r requirements.txt`                                                      |
| `Port 8000 already in use`                       | Use different port: `uvicorn src.main:app --port 8001`                                     |
| `Model download timeout`                         | First API call may take time to download model (~500MB). Wait or check internet connection |
| `Empty text error`                               | Make sure request body has `{"text": "your text here"}`                                    |

## Notes

- The model is preloaded at API startup (and downloaded/cached if needed, ~500MB)
- All predictions are done locally (no external API calls)
- Response times: ~1-2 seconds per request

## License

MIT License - See LICENSE file for details

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from contextlib import asynccontextmanager

from .services import get_classifier, is_model_loaded, predict_sentiment

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_classifier()  # preload on startup
    yield

app = FastAPI(
    title="Simple Sentiment Analysis API",
    description=(
        "API phân tích cảm xúc đa ngôn ngữ sử dụng Hugging Face model "
        "tabularisai/multilingual-sentiment-analysis."
    ),
    version="1.0.0",
    contact={
        "name": "Duong Phan Minh Tri",
        "email": "24120470@student.hcmus.edu.vn",
    },
    docs_url="/docs",
    redoc_url="/redoc",
lifespan=lifespan)


ROOT_RESPONSE_EXAMPLE = {
    "message": "Chào mừng bạn đến với API phân tích cảm xúc đa ngôn ngữ sử dụng Hugging Face, nơi bạn có thể kiểm tra trạng thái hệ thống và dự đoán cảm xúc của văn bản chỉ với vài bước đơn giản."
}

HEALTH_RESPONSE_EXAMPLE = {"status": "healthy", "model_loaded": True}

PREDICT_REQUEST_EXAMPLE = {"text": "Tôi rất yêu thích sản phẩm này!"}

PREDICT_RESPONSE_EXAMPLE = {
    "input": "Tôi rất yêu thích sản phẩm này!",
    "label": "Very Positive",
    "confidence": 0.9587,
}

VALIDATION_ERROR_EXAMPLE = {
    "detail": [
        {
            "loc": ["body", "text"],
            "msg": "Input should be a valid string",
            "type": "string_type",
            "input": 123,
            "ctx": {},
        }
    ]
}

class TextRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": PREDICT_REQUEST_EXAMPLE,
        }
    )

    text: str = Field(
        ...,
        description="Đoạn văn bản cần phân tích cảm xúc",
        examples=["Tôi rất yêu thích sản phẩm này!"],
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": PREDICT_RESPONSE_EXAMPLE}
    )

    input: str = Field(..., description="Văn bản đầu vào")
    label: str = Field(..., description="Nhãn cảm xúc dự đoán")
    confidence: float = Field(..., ge=0, le=1, description="Độ tin cậy từ 0 đến 1")


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": HEALTH_RESPONSE_EXAMPLE}
    )

    status: str
    model_loaded: bool


class RootResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": ROOT_RESPONSE_EXAMPLE}
    )

    message: str


@app.get(
    "/",
    summary="Giới thiệu ngắn gọn về hệ thống",
    description="Trả về thông điệp chào mừng và giới thiệu ngắn gọn về hệ thống.",
    response_model=RootResponse,
    responses={
        200: {
            "description": "Thông tin giới thiệu hệ thống",
            "content": {
                "application/json": {
                    "example": ROOT_RESPONSE_EXAMPLE,
                }
            },
        }
    },
)
def read_root():
    return ROOT_RESPONSE_EXAMPLE


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra sức khỏe hệ thống",
    description="Trả về trạng thái API và cho biết model đã được load hay chưa.",
    responses={
        200: {
            "description": "Trạng thái hệ thống",
            "content": {
                "application/json": {
                    "example": HEALTH_RESPONSE_EXAMPLE,
                }
            },
        }
    },
)
def health_check():
    return {"status": "healthy", "model_loaded": is_model_loaded()}

def preload_model():
    get_classifier()

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Dự đoán cảm xúc",
    description=(
        "Nhận vào text và trả về nhãn cảm xúc cùng độ tin cậy. Các nhãn có thể gồm: `Very Negative`/`Negative`/`Neutral`/`Positive`/`Very Positive`."
    ),
    responses={
        200: {
            "description": "Kết quả dự đoán cảm xúc",
            "content": {
                "application/json": {
                    "example": PREDICT_RESPONSE_EXAMPLE,
                }
            },
        },
        422: {
            "description": "Lỗi validation khi body không đúng schema",
            "content": {
                "application/json": {
                    "example": VALIDATION_ERROR_EXAMPLE,
                }
            },
        },
        400: {"description": "Văn bản rỗng hoặc chỉ toàn khoảng trắng"},
        500: {"description": "Lỗi nội bộ khi dự đoán"},
    },
)
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Văn bản không được để trống")

    try:
        prediction = predict_sentiment(request.text)
        return {
            "input": request.text,
            "label": prediction["label"],
            "confidence": round(float(prediction["score"]), 4),
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")
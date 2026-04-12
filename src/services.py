# Import thư viện pipeline từ transformers để sử dụng mô hình phân tích cảm xúc đa ngôn ngữ
from transformers import pipeline

# Định nghĩa model ID của mô hình phân tích cảm xúc đa ngôn ngữ mà chúng ta sẽ sử dụng
model_id = "tabularisai/multilingual-sentiment-analysis"
_classifier = None

# Hàm để lấy classifier, sử dụng lazy-loading để tránh việc tải mô hình nặng ngay khi khởi động ứng dụng,
# giúp cải thiện khả năng chịu lỗi và giảm thời gian khởi động
def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("sentiment-analysis", model=model_id)
    return _classifier

# Hàm kiểm tra xem model đã được load hay chưa, trả về True nếu model đã được load, ngược lại trả về False
def is_model_loaded() -> bool:
    return _classifier is not None

# Hàm dự đoán cảm xúc từ văn bản đầu vào, sử dụng classifier đã được load để phân tích cảm xúc và trả về kết quả
def predict_sentiment(text: str):
    # Model trả về các label như: Very Negative, Negative, Neutral, Positive, Very Positive
    classifier = get_classifier()
    result = classifier(text)
    return result[0]
from MLP_Predict import MLP_Predict 
import numpy as np

# Đường dẫn tới mô hình và label encoder
model_path = "MLP.pth"
label_encoder_path = "label_encoder.pkl"
image_path = "training_images\sacmt_segment_0080.png"  # Thay bằng đường dẫn thật đến ảnh bạn muốn test
p_mean = 90                # Thay bằng giá trị thật của P_mean

# Khởi tạo bộ dự đoán
predictor = MLP_Predict(model_path, label_encoder_path, image_size=32)

# Dự đoán từ ảnh
predicted_label = predictor.predict(image_path, p_mean)

print(f"Dự đoán: {predicted_label}")

from MLP_Predict import MLP_Predict 
import numpy as np

# Đường dẫn tới mô hình và label encoder
model_path = "MLP.pth"
label_encoder_path = "MLP_label_encoder.pkl"
image_path = r"training_images\null_segment_0035.png"  # Thay bằng đường dẫn thật đến ảnh bạn muốn test
p_mean = 20                # Thay bằng giá trị thật của P_mean

# Khởi tạo bộ dự đoán
predictor = MLP_Predict(model_path, label_encoder_path, image_size=32)

# Dự đoán từ ảnh
predicted_label = predictor.predict(image_path, p_mean)

print(f"Dự đoán: {predicted_label}")

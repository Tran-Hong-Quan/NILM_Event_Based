import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import (
    plot_to_bw_image_with_gaussian_dots,
    flip_ui_image,
    calc_prms
)
import os
from PIL import Image

# --- Cấu hình ---
sampling_rate = 1000
frequency = 50
samples_per_cycle = sampling_rate // frequency

mode = 2  # 1 = chỉ vẽ biểu đồ, 2 = xuất dữ liệu huấn luyện
device_label = "null"  # Nhãn thiết bị (bạn chỉnh lại theo loại thiết bị)
overwrite_mode = False   # True = ghi đè file CSV, False = ghi thêm nếu đã tồn tại

# --- Đọc file CVDs ---
data = np.load("CVDs/all_cycles.npz", allow_pickle=True)
U_CVDS = data["U_CVDS"]
I_CVDS = data["I_CVDS"]

# --- Chuẩn bị lưu dữ liệu nếu ở mode 2 ---
if mode == 2:
    output_folder = "training_images"
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = "MLP_data.csv"

    if overwrite_mode or not os.path.exists(output_csv_path):
        df_output = pd.DataFrame(columns=["segment_id", "label", "P_mean", "image_path"])
        starting_id = 1
    else:
        df_output = pd.read_csv(output_csv_path)
        starting_id = df_output["segment_id"].max() + 1 if not df_output.empty else 1

# --- Lặp qua từng mẫu trong CVDs ---
for i, (U_CVD, I_CVD) in enumerate(zip(U_CVDS, I_CVDS)):
    # --- Tính công suất trung bình ---
    P_mean = calc_prms(U_CVD, I_CVD)

    # --- Sinh ảnh Gaussian ---
    img = plot_to_bw_image_with_gaussian_dots(U_CVD, I_CVD, 32, 32, 2, 0.3)
    img = flip_ui_image(img)

    if mode == 1:
        # Vẽ minh họa
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(U_CVD, I_CVD, color='blue', label=f'Segment {i+1}')
        ax1.set_xlabel("Voltage U (V)")
        ax1.set_ylabel("Current I (A)")
        ax1.set_title(f"I theo U\nP_mean={P_mean:.2f} W")
        ax1.legend()
        ax1.grid(True)

        ax2.imshow(img, cmap='gray')
        ax2.set_title("Ảnh Gaussian")
        ax2.axis("off")

        plt.suptitle(f"Phân tích đoạn {i+1}", fontsize=14)
        plt.tight_layout()
        plt.show()

    elif mode == 2:
        # Lưu ảnh + dữ liệu
        global_id = starting_id + i
        image_filename = f"{device_label}_segment_{global_id:04d}.png"
        image_path = os.path.join(output_folder, image_filename)
        Image.fromarray(img).save(image_path)

        df_output.loc[len(df_output)] = {
            "segment_id": global_id,
            "label": device_label,
            "P_mean": round(P_mean, 4),
            "image_path": image_path
        }

# --- Ghi file CSV nếu ở mode 2 ---
if mode == 2:
    df_output.to_csv(output_csv_path, index=False)
    print(f"✅ Đã lưu dữ liệu vào: {output_csv_path}")

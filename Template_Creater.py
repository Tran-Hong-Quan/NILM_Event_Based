import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import CycleInterpolator, close_curve, plot_to_bw_image_with_gaussian_dots
import os
import csv
import time
import json  # để lưu mảng U/I
import config

# --- Chọn chế độ ---
mode = 1 #0 là chỉ ảnh, 1 là cơ lưu vào file
writeMode = "a"  # a = thêm, w = ghi đè toàn bộ file CSV

# --- Cấu hình ---
csv_path = r"ElectricDatas\MyData\New\data csv\NO\quat_event_no.csv"
parts = csv_path.replace("\\", "/").split("/")
csv_path = os.path.join(*parts)

sampling_rate = 1000
frequency = 50
samples_per_cycle = sampling_rate // frequency
interp_factor = 10

# --- Chọn start và end ---
start_cycle = 1000
end_cycle = 5000

# --- Nhãn và sai số ---
label = "quat"
p_tolerance = 0.7

# --- Đọc dữ liệu ---
df = pd.read_csv(csv_path)
time_axis = np.arange(len(df)) / sampling_rate
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values

# Giới hạn start/end
total_cycles = len(df) // samples_per_cycle
end_cycle = min(end_cycle, total_cycles)

start_idx = start_cycle * samples_per_cycle
end_idx = end_cycle * samples_per_cycle

# --- Tính đường cong I–U ---
I_seg = I_raw[start_idx:end_idx]
U_seg = U_raw[start_idx:end_idx]

interp = CycleInterpolator(samples_per_cycle, interp_factor)
interp.update_batch(I_seg, U_seg)
U, I = interp.get_average()

P_mean = np.mean(Power[start_idx:end_idx])

# --- Sinh ảnh Gaussian (chỉ để hiển thị, không lưu) ---
U_Closed, I_Closed = close_curve(U, I)
img = plot_to_bw_image_with_gaussian_dots(U_Closed, I_Closed, config.IMAGE_SIZE, config.IMAGE_SIZE,config.IMG_DOT_RADIUS,config.IMG_DOT_RADIUS)

# --- Vẽ ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].plot(time_axis, Power, label="Power (W)", color='black')
axes[0].axvline(time_axis[start_idx], color='blue', linestyle='--', label="Start")
axes[0].axvline(time_axis[end_idx-1], color='blue', linestyle='--', label="End")
axes[0].fill_between(time_axis[start_idx:end_idx], Power[start_idx:end_idx], alpha=0.2, color='blue')
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Power (W)")
axes[0].set_title("Power theo thời gian")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(U_Closed, I_Closed, label=f'Test (Pmean={P_mean:.2f} W)', color='blue')
axes[1].set_xlabel("Voltage U (V)")
axes[1].set_ylabel("Current I (A)")
axes[1].set_title("Trung bình I theo U")
axes[1].grid(True)
axes[1].legend()

axes[2].imshow(img, cmap='gray')
axes[2].set_title("Ảnh I–U Gaussian")
axes[2].axis('off')
plt.tight_layout()
plt.show()

# --- Lưu metadata ---
if mode == 1:
    # File metadata lưu cùng thư mục code
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_csv = os.path.join(script_dir, "Template_data.csv")

    file_exists = os.path.isfile(metadata_csv)

    # Đọc metadata cũ để kiểm tra trùng
    existing_rows = []
    if file_exists:
        with open(metadata_csv, newline='', encoding='utf-8') as f:
            existing_rows = list(csv.reader(f))

    # Tạo hàng mới
    new_row = [f"{label}_{int(time.time()*1000)}", f"{P_mean:.3f}", label, p_tolerance]

    if new_row not in existing_rows:
        with open(metadata_csv, mode=writeMode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["id", "P_mean", "label", "p_tolerance"])
            writer.writerow(new_row)
        print(f"[Saved] P_mean={P_mean:.3f} | Label={label} saved in Template_data.csv")
    else:
        print("[Skipped] Dữ liệu này đã tồn tại trong Template_data.csv")

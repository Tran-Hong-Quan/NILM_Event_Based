import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import CycleInterpolator, close_curve, plot_to_bw_image_with_gaussian_dots, flip_ui_image
import os
from PIL import Image

# --- Cấu hình ---
csv_path = r"ElectricDatas\MyData\New\data csv\NO\tulanh_event_no.csv"
parts = csv_path.replace("\\", "/").split("/")
csv_path = os.path.join(*parts)

sampling_rate = 1000
frequency = 50
samples_per_cycle = sampling_rate // frequency
test_cycles = 100
interp_factor = 10

mode = 2  # 1 = chỉ vẽ biểu đồ, 2 = xuất dữ liệu huấn luyện
device_label = "tulanh"  # Nhãn thiết bị
overwrite_mode = False   # True = ghi đè file CSV, False = ghi thêm nếu đã tồn tại

# --- Đọc dữ liệu ---
df = pd.read_csv(csv_path)
time = np.arange(len(df)) / sampling_rate
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values

total_samples = len(df)
step_size = test_cycles * samples_per_cycle

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

# --- Lặp qua từng đoạn ---
for i, start in enumerate(range(0, total_samples - step_size + 1, step_size)):
    end = start + step_size
    I_seg = I_raw[start:end]
    U_seg = U_raw[start:end]
    Power_seg = Power[start:end]

    # --- Tính công suất trung bình ---
    P_mean = abs(np.mean(U_seg * I_seg))

    # --- Nội suy và tạo ảnh ---
    interp = CycleInterpolator(samples_per_cycle, interp_factor)
    interp.update_batch(I_seg, U_seg)
    U_avg, I_avg = interp.get_average()
    U_closed, I_closed = close_curve(U_avg, I_avg)
    #print(len(U_closed))
    img = plot_to_bw_image_with_gaussian_dots(U_closed, I_closed, 32, 32, 2, 0.3)
    img = flip_ui_image(img)

    if mode == 1:
        # --- Vẽ biểu đồ ---
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Hình 1: Power
        axs[0].plot(time, Power, color='black', label="Power (W)")
        axs[0].axvline(time[start], color='blue', linestyle='--', label=f"Start {i+1}")
        axs[0].axvline(time[end], color='blue', linestyle='--', label=f"End {i+1}")
        axs[0].fill_between(time[start:end], Power[start:end], alpha=0.2, color='blue')
        axs[0].text((time[start] + time[end]) / 2, np.max(Power)*0.8,
                    f"P_mean = {P_mean:.2f} W", color='red', ha='center')
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Power (W)")
        axs[0].set_title("Power theo thời gian")
        axs[0].legend()
        axs[0].grid(True)

        # Hình 2: I-U
        axs[1].plot(U_closed, I_closed, color='blue', label=f'Test {i+1}')
        axs[1].set_xlabel("Voltage U (V)")
        axs[1].set_ylabel("Current I (A)")
        axs[1].set_title("I theo U")
        axs[1].legend()
        axs[1].grid(True)

        # Hình 3: Ảnh Gaussian
        axs[2].imshow(img, cmap='gray')
        axs[2].set_title("Ảnh Gaussian")
        axs[2].axis("off")

        fig.suptitle(f"Phân tích đoạn {i+1} - P_mean = {P_mean:.2f} W", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

    elif mode == 2:
        # --- Ghi ảnh và dữ liệu ---
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

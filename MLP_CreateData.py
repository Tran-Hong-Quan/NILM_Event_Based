import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import (
    CycleInterpolator, close_curve, plot_to_bw_image_with_gaussian_dots,
    flip_ui_image, calc_prms, align_phase
)
import os
from PIL import Image
import config

# --- Cấu hình ---
csv_path = r"ElectricDatas\MyNewData\NO\quat_event_no.csv"
csv_path = os.path.normpath(csv_path)

sampling_rate = 1000
frequency = 50
samples_per_cycle = sampling_rate // frequency
test_cycles = 100
interp_factor = 10

MODE = 1  # 1 = vẽ biểu đồ, 2 = xuất dữ liệu huấn luyện
CREATE_CVDS = True
DEVICE_LABEL = "tulanh"
OVERWRITE_MODE = False # Nên là false

# --- Đọc dữ liệu ---
df = pd.read_csv(csv_path)
time = np.arange(len(df)) / sampling_rate
Power, I_raw, U_raw = df["Power"].values, df["In"].values, df["Un"].values

total_samples = len(df)
step_size = test_cycles * samples_per_cycle

# --- Chuẩn bị CSV nếu mode = 2 ---
if MODE == 2:
    output_folder = "training_images"
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = "MLP_data.csv"

    if OVERWRITE_MODE or not os.path.exists(output_csv_path):
        df_output = pd.DataFrame(columns=["segment_id", "label", "P_mean", "image_path"])
        starting_id = 1
    else:
        df_output = pd.read_csv(output_csv_path)
        starting_id = df_output["segment_id"].max() + 1 if not df_output.empty else 1

# --- Hàm vẽ ---
def plot_results(i, P_mean, U_curve, I_curve, img, start, end):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Power
    axs[0].plot(time, Power, color='black', label="Power (W)")
    axs[0].axvline(time[start], color='blue', linestyle='--')
    axs[0].axvline(time[end], color='blue', linestyle='--')
    axs[0].fill_between(time[start:end], Power[start:end], alpha=0.2, color='blue')
    axs[0].text((time[start] + time[end]) / 2, np.max(Power)*0.8,
                f"P_mean = {P_mean:.2f} W", color='red', ha='center')
    axs[0].set_title("Power theo thời gian"); axs[0].legend(); axs[0].grid(True)

    # I-U
    axs[1].plot(U_curve, I_curve, color='blue', label=f'Test {i+1}')
    axs[1].set_xlabel("Voltage U (V)"); axs[1].set_ylabel("Current I (A)")
    axs[1].set_title("I theo U"); axs[1].legend(); axs[1].grid(True)

    # Gaussian
    axs[2].imshow(img, cmap='gray'); axs[2].set_title("Ảnh Gaussian"); axs[2].axis("off")

    fig.suptitle(f"Phân tích đoạn {i+1} - P_mean = {P_mean:.2f} W", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

# --- Hàm lưu ---
def save_result(i, P_mean, img, starting_id):
    global df_output
    global_id = starting_id + i
    image_filename = f"{DEVICE_LABEL}_segment_{global_id:04d}.png"
    image_path = os.path.join("training_images", image_filename)
    Image.fromarray(img).save(image_path)

    df_output.loc[len(df_output)] = {
        "segment_id": global_id,
        "label": DEVICE_LABEL,
        "P_mean": round(P_mean, 4),
        "image_path": image_path
    }

# --- Chạy ---
if not CREATE_CVDS:
    # Chạy trên dữ liệu gốc
    for i, start in enumerate(range(0, total_samples - step_size + 1, step_size)):
        end = start + step_size
        I_seg, U_seg = I_raw[start:end], U_raw[start:end]

        P_mean = abs(np.mean(U_seg * I_seg))

        interp = CycleInterpolator(samples_per_cycle, interp_factor)
        interp.update_batch(I_seg, U_seg)
        U_avg, I_avg = interp.get_average()
        U_closed, I_closed = close_curve(U_avg, I_avg)

        img = flip_ui_image(plot_to_bw_image_with_gaussian_dots(U_closed, I_closed,config.IMAGE_SIZE, config.IMAGE_SIZE,config.IMG_DOT_RADIUS,config.IMG_DOT_RADIUS))

        if MODE == 1:
            plot_results(i, P_mean, U_closed, I_closed, img, start, end)
        else:
            save_result(i, P_mean, img, starting_id)
else:
    # Chạy trên dữ liệu CVDs đã lưu
    data = np.load("CVDs/all_cycles.npz", allow_pickle=True)
    U_CVDS, I_CVDS = data["U_CVDS"], data["I_CVDS"]

    start, end = 20000, 20000 + step_size*10
    I_seg, U_seg = I_raw[start:end], U_raw[start:end]

    interp = CycleInterpolator(samples_per_cycle, interp_factor)
    interp.update_batch(I_seg, U_seg)
    U_avg, I_avg = interp.get_average()

    for i, (U_CVD, I_CVD) in enumerate(zip(U_CVDS, I_CVDS)):
        U_CVD, best_shift = align_phase(U_avg, U_CVD)
        I_CVD = np.roll(I_CVD, -int(best_shift))

        U_AVG, I_AVG = U_avg.copy(), I_avg.copy()
        I_AVG += I_CVD

        img = flip_ui_image(plot_to_bw_image_with_gaussian_dots(U_AVG, I_AVG, config.IMAGE_SIZE, config.IMAGE_SIZE,config.IMG_DOT_RADIUS,config.IMG_DOT_RADIUS))
        P_mean = calc_prms(U_AVG, I_AVG)

        if MODE == 1:
            plot_results(i, P_mean, U_AVG, I_AVG, img, start, end)
        else:
            save_result(i, P_mean, img, starting_id)

# --- Xuất CSV nếu cần ---
if MODE == 2:
    df_output.to_csv(output_csv_path, index=False)
    print(f"✅ Đã lưu dữ liệu vào: {output_csv_path}")

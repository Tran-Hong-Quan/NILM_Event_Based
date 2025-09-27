import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import (
    CycleInterpolator, align_phase, calc_prms, 
    plot_to_bw_image_with_gaussian_dots, flip_ui_image, is_right_side_greater
    ,close_curve
)
import os
from DrawUIImage import plt_ui_full_onefig
import config

# --- Cấu hình ---
maxCvdCounts = 100
csv_path = r"ElectricDatas\MyNewData\NO\sacmt_event_no.csv"
parts = csv_path.replace("\\", "/").split("/")
csv_path = os.path.join(*parts)
sampling_rate = 1000
frequency = 50
samples_per_cycle = sampling_rate // frequency
test_cycles = 100
interp_factor = 10
SAMPLE_PER_IMAGE = test_cycles * samples_per_cycle

# --- Đọc dữ liệu ---
df = pd.read_csv(csv_path)
time = np.arange(len(df)) / sampling_rate
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values
print("Số mẫu dữ liệu:", len(I_raw))

# Số chu kỳ tổng trong dữ liệu
total_cycles = len(I_raw) // samples_per_cycle

# --- Chuẩn bị lưu tất cả dữ liệu ---
U_CVDS = []
I_CVDS = []

stop = False

# --- Quét toàn bộ dữ liệu ---
for delay1 in range(0, total_cycles - test_cycles, test_cycles):
    for delay2 in range(delay1 + test_cycles, total_cycles - test_cycles, test_cycles):

        # --- Xử lý test 1 ---
        start1 = delay1 * samples_per_cycle
        end1 = start1 + SAMPLE_PER_IMAGE
        interp1 = CycleInterpolator(samples_per_cycle, interp_factor)
        interp1.update_batch(I_raw[start1:end1], U_raw[start1:end1])
        U1, I1 = interp1.get_average()

        # --- Xử lý test 2 ---
        start2 = delay2 * samples_per_cycle
        end2 = start2 + SAMPLE_PER_IMAGE
        interp2 = CycleInterpolator(samples_per_cycle, interp_factor)
        interp2.update_batch(I_raw[start2:end2], U_raw[start2:end2])
        U2, I2 = interp2.get_average()

        # --- Căn chỉnh pha ---
        U1_aligned, best_shift = align_phase(U2, U1)
        I1_aligned = np.roll(I1, -best_shift)
        I_diff = (I2 - I1_aligned)
        I_diff *= is_right_side_greater(I_diff,U2)
        

        # --- Tính toán ---
        delta_P_mean = calc_prms(U2,I_diff)
        print(delta_P_mean)
        if delta_P_mean > 7:
            img = plot_to_bw_image_with_gaussian_dots(U2, I_diff, config.IMAGE_SIZE, config.IMAGE_SIZE,config.IMG_DOT_RADIUS,config.IMG_DOT_ALPHA)
            img = flip_ui_image(img)
            #--- Vẽ / Lưu ---
            # plt_ui_full_onefig(
            #     sampling_rate, Power,
            #     start1, end1, start2, end2,
            #     U1, I1, U2, I2, I_diff,
            #     img, delta_P_mean,
            #     "none", 0
            # )
            
            U_CVDS.append(U2)
            I_CVDS.append(I_diff)
            print("Số cvds = "  + str(len(U_CVDS)))
            if len(U_CVDS) >= maxCvdCounts:
                stop = True
                break
    if stop:
        break
            
# --- Lưu tất cả vào 1 file npz ---
os.makedirs("CVDs", exist_ok=True)
np.savez("CVDs/all_cycles.npz", U_CVDS=U_CVDS, I_CVDS=I_CVDS)

    

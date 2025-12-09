import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import (
    CycleInterpolator, align_phase, close_curve, close_array,
    calc_prms, plot_to_bw_image_with_gaussian_dots,
    smooth_savgol, is_right_side_greater
)
import os
import config

# --- Cấu hình ---
csv_path = r"ElectricDatas\MyData\data csv 2\quat_event_sacmt.csv"
parts = csv_path.replace("\\", "/").split("/")
csv_path = os.path.join(*parts)

sampling_rate = 1000
frequency = 50
samples_per_cycle = sampling_rate // frequency
test_cycles = 100
interp_factor = 10

delay1 = 62 * frequency
delay2 = 68 * frequency

# --- Đọc dữ liệu ---
df = pd.read_csv(csv_path)
time = np.arange(len(df)) / sampling_rate
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values

# --- Test 1 ---
start1 = delay1 * samples_per_cycle
end1   = start1 + test_cycles * samples_per_cycle

interp1 = CycleInterpolator(samples_per_cycle, interp_factor)
for i in range(test_cycles):
    s = i * samples_per_cycle
    e = (i + 1) * samples_per_cycle
    interp1.update(I_raw[start1+s:start1+e], U_raw[start1+s:start1+e])
U1, I1 = interp1.get_average()

# --- Test 2 ---
start2 = delay2 * samples_per_cycle
end2   = start2 + test_cycles * samples_per_cycle

interp2 = CycleInterpolator(samples_per_cycle, interp_factor)
for i in range(test_cycles):
    s = i * samples_per_cycle
    e = (i + 1) * samples_per_cycle
    interp2.update(I_raw[start2+s:start2+e], U_raw[start2+s:start2+e])
U2, I2 = interp2.get_average()

# --- Căn pha ---
U1_aligned, best_shift = align_phase(U2, U1)
I1_aligned = np.roll(I1, -best_shift)
I_diff = (I2 - I1_aligned)

sign = is_right_side_greater(I_diff, U2)
I_diff *= sign
print("Curve Direction =", sign)
print("Delta P_rms =", calc_prms(U1, I1) - calc_prms(U2, I2))

# ================================================================
#  CỬA SỔ 1 — U_raw, I_raw, P theo thời gian (3 subplot chung trục)
# ================================================================
figA, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

ax[0].plot(time, Power, color='black')
ax[0].set_ylabel("P (W)")
ax[0].set_title("Power – Voltage – Current theo thời gian (đồng bộ trục)")

ax[1].plot(time, U_raw, color='black')
ax[1].set_ylabel("U_raw (V)")

ax[2].plot(time, I_raw, color='black')
ax[2].set_ylabel("I_raw (A)")
ax[2].set_xlabel("Time (s)")

# Highlight vùng test
for a in ax:
    # Test 1
    a.axvline(time[start1], color='blue', linestyle='--')
    a.axvline(time[end1],   color='blue', linestyle='--')
    a.fill_between(time[start1:end1], a.get_ylim()[0], a.get_ylim()[1],
                   color='blue', alpha=0.15)

    # Test 2
    a.axvline(time[start2], color='red', linestyle='--')
    a.axvline(time[end2],   color='red', linestyle='--')
    a.fill_between(time[start2:end2], a.get_ylim()[0], a.get_ylim()[1],
                   color='red', alpha=0.15)

plt.tight_layout()

# ================================================================
#  CỬA SỔ 2 — Test1, Test2, ΔI, Ảnh Gaussian (gộp chung 2×2)
# ================================================================
figB, axs = plt.subplots(2, 2, figsize=(12, 10))

# --- Test 1 ---
U1_closed, I1_closed = close_curve(U1, I1)
axs[0, 0].plot(U1_closed, I1_closed, color='blue')
axs[0, 0].set_title("Test 1 – U–I")
axs[0, 0].set_xlabel("U (V)")
axs[0, 0].set_ylabel("I (A)")
axs[0, 0].grid(True)

# --- Test 2 ---
U2_closed, I2_closed = close_curve(U2, I2)
axs[0, 1].plot(U2_closed, I2_closed, color='red')
axs[0, 1].set_title("Test 2 – U–I")
axs[0, 1].set_xlabel("U (V)")
axs[0, 1].set_ylabel("I (A)")
axs[0, 1].grid(True)

# --- I_diff ---
I_diff_closed = close_array(I_diff)
U_smooth = smooth_savgol(U2_closed, window_length=21, polyorder=5)
I_smooth = smooth_savgol(I_diff_closed, window_length=21, polyorder=5)
axs[1, 0].plot(U_smooth, I_smooth, color='purple')
axs[1, 0].set_title("ΔI = I2 – I1_aligned (smooth)")
axs[1, 0].set_xlabel("U (V)")
axs[1, 0].set_ylabel("ΔI (A)")
axs[1, 0].grid(True)

# --- Ảnh Gaussian ---
img = plot_to_bw_image_with_gaussian_dots(
    U2_closed, I_diff_closed,
    config.IMAGE_SIZE, config.IMAGE_SIZE,
    config.IMG_DOT_RADIUS, config.IMG_DOT_ALPHA
)
axs[1, 1].imshow(img, cmap='gray')
axs[1, 1].set_title("Ảnh Gaussian – ΔI")
axs[1, 1].axis("off")

plt.tight_layout()
plt.show()

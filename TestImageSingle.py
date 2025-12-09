import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import CycleInterpolator, close_curve, plot_to_bw_image_with_gaussian_dots, flip_ui_image
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

delay1 = 500

# --- Đọc dữ liệu ---
df = pd.read_csv(csv_path)
time = np.arange(len(df)) / sampling_rate
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values

# --- Xử lý test 1 ---
start1 = delay1 * samples_per_cycle
end1 = start1 + test_cycles * samples_per_cycle
I_seg1 = I_raw[start1:end1]
U_seg1 = U_raw[start1:end1]

interp1 = CycleInterpolator(samples_per_cycle, interp_factor)
interp1.update_batch(I_seg1, U_seg1)
U, I = interp1.get_average()

# --- Tính P trung bình ---
P_mean = np.mean(U * I)

# ============================================================
#  CỬA SỔ 1: VẼ P, U_raw, I_raw THEO THỜI GIAN — CÙNG TRỤC X
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# P
axes[0].plot(time, Power, color='black')
axes[0].set_ylabel("P (W)")
axes[0].set_title("P, U_raw, I_raw theo thời gian (đồng bộ trục X)")
axes[0].grid(True)

# U
axes[1].plot(time, U_raw, color='black')
axes[1].set_ylabel("U (V)")
axes[1].grid(True)

# I
axes[2].plot(time, I_raw, color='black')
axes[2].set_ylabel("I (A)")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True)

# Highlight vùng Test 1
for ax in axes:
    ax.axvline(time[start1], color='blue', linestyle='--')
    ax.axvline(time[end1], color='blue', linestyle='--')
    ax.fill_between(time[start1:end1],
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                    color='blue', alpha=0.15)

plt.tight_layout()

# ============================================================
#  CỬA SỔ 2: TEST + ẢNH — GỘP CHUNG
# ============================================================

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# --- Test 1: U-I Closed Curve ---
U_Closed, I_Closed = close_curve(U, I)
ax1.plot(U_Closed, I_Closed, color='red', label=f'P_mean = {abs(P_mean):.2f} W')
ax1.set_xlabel("Voltage U (V)")
ax1.set_ylabel("Current I (A)")
ax1.set_title("Đường U-I trung bình (Test 1)")
ax1.grid(True)
ax1.legend()

# --- Ảnh Gaussian ---
img = plot_to_bw_image_with_gaussian_dots(
    U_Closed, I_Closed,
    config.IMAGE_SIZE, config.IMAGE_SIZE,
    config.IMG_DOT_RADIUS, config.IMG_DOT_ALPHA
)
img = flip_ui_image(img)

ax2.imshow(img, cmap='gray')
ax2.set_title(f"Ảnh Gaussian (P_mean = {P_mean:.2f} W)")
ax2.axis("off")

plt.tight_layout()
plt.show()

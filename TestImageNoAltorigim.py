import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import CycleInterpolator, close_curve, plot_to_bw_image_with_gaussian_dots, flip_ui_image
import os
import config

# --- Cấu hình ---
csv_path = r"ElectricDatas\MyData\data csv 2\NO\sacmt_event_no.csv"
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

U = U_seg1
I = I_seg1

# --- Tính P trung bình ---
P_mean = np.mean(U * I)

# ============================================================
#  VẼ FULL TIME U, I, P CHUNG 1 CỬA SỔ — CÙNG TRỤC THỜI GIAN
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# U
axes[0].plot(time, U_raw, color='black')
axes[0].set_ylabel("U (V)")
axes[0].grid(True)
axes[0].set_title("U, I, P theo thời gian (đồng bộ trục thời gian)")

# I
axes[1].plot(time, I_raw, color='black')
axes[1].set_ylabel("I (A)")
axes[1].grid(True)

# P
axes[2].plot(time, Power, color='black')
axes[2].set_ylabel("P (W)")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True)

# Đánh dấu vùng test giống hình cũ
for ax in axes:
    ax.axvline(time[start1], color='blue', linestyle='--')
    ax.axvline(time[end1], color='blue', linestyle='--')
    ax.fill_between(time[start1:end1],
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                    alpha=0.15, color='blue')

plt.tight_layout()

# ============================================================
#  VẼ TEST 
# ============================================================

plt.figure(figsize=(6, 6))
plt.scatter(U, I, label=f'P_mean = {abs(P_mean):.2f} W', color='blue')
plt.xlabel("Voltage U (V)")
plt.ylabel("Current I (A)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

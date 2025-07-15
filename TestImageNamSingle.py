import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import CycleInterpolator, close_curve
import os

# --- Cấu hình ---
sv_path = os.path.join("ElectricDatas", "MyData", "data csv", "mayep_maysay_tulanh_event_sacmt.csv")
sampling_rate = 1000
frequency = 50
samples_per_cycle = sampling_rate // frequency
test_cycles = 40
interp_factor = 10

delay1 = 500

# --- Đọc dữ liệu ---
df = pd.read_csv(sv_path)
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

# --- VẼ HÌNH 1: Power theo thời gian ---
plt.figure(figsize=(12, 5))
plt.plot(time, Power, label="Power (W)", color='black')

plt.axvline(time[start1], color='blue', linestyle='--', label="Test 1 Start")
plt.axvline(time[end1], color='blue', linestyle='--', label="Test 1 End")
plt.fill_between(time[start1:end1], Power[start1:end1], alpha=0.2, color='blue')

plt.text(time[start1], np.max(Power)*0.95, "Start 1", color='blue')
plt.text(time[end1], np.max(Power)*0.95, "End 1", color='blue', ha='right')

plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.title("Power theo thời gian (đánh dấu 2 vùng test)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- VẼ HÌNH 2: Test 1 ---
plt.figure(figsize=(6, 6))
U_Closed, I_Closed = close_curve(U, I)
plt.plot(U_Closed, I_Closed, label='Test 1', color='blue')
plt.xlabel("Voltage U (V)")
plt.ylabel("Current I (A)")
plt.title("Trung bình I theo U (Test 1)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

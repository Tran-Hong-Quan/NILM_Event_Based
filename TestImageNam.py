import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import CycleInterpolator, align_phase, close_curve, close_array, calc_prms, plot_to_bw_image, smooth_savgol, is_right_side_greater 
# --- Cấu hình ---
csv_path = "ElectricDatas\MyData\data csv\mayep_maysay_tulanh_event_sacmt.csv"
sampling_rate = 1000
frequency = 50
samples_per_cycle = sampling_rate // frequency
test_cycles = 50
interp_factor = 6

delay1 = 60 * frequency
delay2 = 80 * frequency

# --- Đọc dữ liệu ---
df = pd.read_csv(csv_path)
time = np.arange(len(df)) / sampling_rate
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values
print(len(I_raw))

# --- Xử lý test 1 ---
start1 = delay1 * samples_per_cycle
end1 = start1 + test_cycles * samples_per_cycle
I_seg1 = I_raw[start1:end1]
U_seg1 = U_raw[start1:end1]

interp1 = CycleInterpolator(samples_per_cycle, interp_factor)
for i in range(test_cycles):
    s = i * samples_per_cycle
    e = (i + 1) * samples_per_cycle
    interp1.update(I_seg1[s:e], U_seg1[s:e])
U1, I1 = interp1.get_average()

# --- Xử lý test 2 ---
start2 = delay2 * samples_per_cycle
end2 = start2 + test_cycles * samples_per_cycle
I_seg2 = I_raw[start2:end2]
U_seg2 = U_raw[start2:end2]

interp2 = CycleInterpolator(samples_per_cycle, interp_factor)
for i in range(test_cycles):
    s = i * samples_per_cycle
    e = (i + 1) * samples_per_cycle
    interp2.update(I_seg2[s:e], U_seg2[s:e])
U2, I2 = interp2.get_average()

U1_aligned, best_shift = align_phase(U2, U1)
I1_aligned = np.roll(I1, -best_shift)
I_diff = (I2 - I1_aligned)

sign = is_right_side_greater(I_diff,U2)
I_diff *= sign
print("Curve Direction = " + str(sign))
print("Delta P_rms = " + str(calc_prms(U1,I1)-calc_prms(U2,I2)))

# --- VẼ HÌNH 1: Power theo thời gian ---
plt.figure(figsize=(12, 5))
plt.plot(time, Power, label="Power (W)", color='black')

plt.axvline(time[start1], color='blue', linestyle='--', label="Test 1 Start")
plt.axvline(time[end1], color='blue', linestyle='--', label="Test 1 End")
plt.fill_between(time[start1:end1], Power[start1:end1], alpha=0.2, color='blue')

plt.axvline(time[start2], color='red', linestyle='--', label="Test 2 Start")
plt.axvline(time[end2], color='red', linestyle='--', label="Test 2 End")
plt.fill_between(time[start2:end2], Power[start2:end2], alpha=0.2, color='red')

plt.text(time[start1], np.max(Power)*0.95, "Start 1", color='blue')
plt.text(time[end1], np.max(Power)*0.95, "End 1", color='blue', ha='right')
plt.text(time[start2], np.max(Power)*0.9, "Start 2", color='red')
plt.text(time[end2], np.max(Power)*0.9, "End 2", color='red', ha='right')

plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.title("Power theo thời gian (đánh dấu 2 vùng test)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- VẼ HÌNH 2: Test 1 ---
plt.figure(figsize=(6, 6))
U1_closed,I1_closed = close_curve(U1,I1)
plt.plot(U1_closed, I1_closed, label='Test 1', color='blue')
plt.xlabel("Voltage U (V)")
plt.ylabel("Current I (A)")
plt.title("Trung bình I theo U (Test 1)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- VẼ HÌNH 3: Test 2 ---
plt.figure(figsize=(6, 6))
U2_closed,I2_closed = close_curve(U2,I2)
plt.plot(U2_closed, I2_closed, label='Test 2', color='red')
plt.xlabel("Voltage U (V)")
plt.ylabel("Current I (A)")
plt.title("Trung bình I theo U (Test 2)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- VẼ HÌNH 4: I2 - I1 sau khi căn pha ---
plt.figure(figsize=(6, 6))
I_diff_closed = close_array(I_diff)
U_smooth = smooth_savgol(U2_closed, window_length=21, polyorder=5)
I_smooth = smooth_savgol(I_diff_closed, window_length=21, polyorder=5)
plt.plot(U_smooth, I_smooth, label='I2 - I1 (đã căn pha)', color='purple')
plt.xlabel("Voltage U (V)")
plt.ylabel("Current diff (A)")
plt.title("Hiệu I2 - I1 sau khi căn pha theo U")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(6, 6))
img = plot_to_bw_image(U2_closed, I_diff_closed,32,32)
plt.title("Ảnh I2 - I1")
plt.imshow(img, cmap='gray')
plt.tight_layout()

plt.show()

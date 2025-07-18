import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import close_curve, smooth_savgol, close_array, plot_to_bw_image

def plt_ui_full(sampling_rate, Power, start1, end1, start2, end2, U1, I1, U2, I2, I_diff):
    time = np.arange(len(Power)) / sampling_rate
    
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

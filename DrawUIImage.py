import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import close_curve, smooth_savgol, close_array, plot_to_bw_image

def plt_ui_full(sampling_rate, Power, start1, end1, start2, end2, U1, I1, U2, I2, I_diff):
    time = np.arange(len(Power)) / sampling_rate

    # --- VẼ HÌNH 1: Power theo thời gian ---
    plt.figure(figsize=(12, 5))
    plt.plot(time, Power, label="Power (W)", color='black')

    plt.axvline(time[start1], color='blue', linestyle='--', label="Test 1 Start")
    plt.axvline(time[end1], color='blue', linestyle='--')
    plt.fill_between(time[start1:end1], Power[start1:end1], alpha=0.2, color='blue')

    plt.axvline(time[start2], color='red', linestyle='--', label="Test 2 Start")
    plt.axvline(time[end2], color='red', linestyle='--')
    plt.fill_between(time[start2:end2], Power[start2:end2], alpha=0.2, color='red')

    plt.text(time[start1], np.max(Power)*0.95, "Start 1", color='blue')
    plt.text(time[end1], np.max(Power)*0.95, "End 1", color='blue', ha='right')
    plt.text(time[start2], np.max(Power)*0.9, "Start 2", color='red')
    plt.text(time[end2], np.max(Power)*0.9, "End 2", color='red', ha='right')

    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("Power theo thời gian (đánh dấu 2 vùng test)")
    plt.ylim(bottom=Power.min())  # ✅ chỉ hiển thị từ giá trị nhỏ nhất
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- VẼ HÌNH 2: Test 1 ---
    plt.figure(figsize=(6, 6))
    U1_closed, I1_closed = close_curve(U1, I1)
    plt.plot(U1_closed, I1_closed, label='Test 1', color='blue')
    plt.xlabel("Voltage U (V)")
    plt.ylabel("Current I (A)")
    plt.title("Trung bình I theo U (Test 1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # --- VẼ HÌNH 3: Test 2 ---
    plt.figure(figsize=(6, 6))
    U2_closed, I2_closed = close_curve(U2, I2)
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

    # --- VẼ HÌNH 5: Ảnh đen trắng ---
    plt.figure(figsize=(6, 6))
    img = plot_to_bw_image(U2_closed, I_diff_closed, 32, 32)
    plt.title("Ảnh I2 - I1")
    plt.imshow(img, cmap='gray')
    plt.tight_layout()

    plt.show()

    
def plt_event_window(power_array, sampling_rate, start_idx, end_idx, idx):
    import matplotlib.pyplot as plt
    time = np.arange(len(power_array)) / sampling_rate

    plt.figure(figsize=(12, 4))
    plt.plot(time, power_array, color='black', label='Power')

    plt.axvline(x=start_idx / sampling_rate, color='orange', linestyle='--', label='Window Start')
    plt.axvline(x=end_idx / sampling_rate, color='orange', linestyle='--', label='Window End')
    plt.fill_between(time[start_idx:end_idx], power_array[start_idx:end_idx], color='orange', alpha=0.2)

    plt.axvline(x=idx / sampling_rate, color='red', linestyle='--', label='Event Detected')
    plt.title(f"Power with Event Window at idx={idx}")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.ylim(bottom=power_array.min())  # ✅ chỉ hiển thị từ giá trị nhỏ nhất
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plt_ui_full_onefig(sampling_rate, Power, start1, end1, start2, end2, U1, I1, U2, I2, I_diff):
    time = np.arange(len(Power)) / sampling_rate

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    fig.suptitle("So sánh ảnh trước và sau sự kiện", fontsize=16)

    # --- Hàng trên: Power ---
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(time, Power, label="Power (W)", color='black')
    ax_top.axvline(time[start1], color='blue', linestyle='--', label="Start 1")
    ax_top.axvline(time[end1], color='blue', linestyle='--')
    ax_top.fill_between(time[start1:end1], Power[start1:end1], alpha=0.2, color='blue')

    ax_top.axvline(time[start2], color='red', linestyle='--', label="Start 2")
    ax_top.axvline(time[end2], color='red', linestyle='--')
    ax_top.fill_between(time[start2:end2], Power[start2:end2], alpha=0.2, color='red')
    ax_top.set_ylim(bottom=Power.min())
    ax_top.set_title("Power theo thời gian")
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel("Power (W)")
    ax_top.grid(True)
    ax_top.legend()

    # --- Hàng dưới: 4 cột ---
    # 1. Test 2 - hình trừ (chuyển lên trước)
    ax1 = fig.add_subplot(gs[1, 0])
    U2_closed, I2_closed = close_curve(U2, I2)
    ax1.plot(U2_closed, I2_closed, label='Test 2', color='red')
    ax1.set_title("I theo U - Sau (Test 2)")
    ax1.set_xlabel("U (V)")
    ax1.set_ylabel("I (A)")
    ax1.grid(True)
    ax1.legend()

    # 2. Test 1 - hình bị trừ
    ax2 = fig.add_subplot(gs[1, 1])
    U1_closed, I1_closed = close_curve(U1, I1)
    ax2.plot(U1_closed, I1_closed, label='Test 1', color='blue')
    ax2.set_title("I theo U - Trước (Test 1)")
    ax2.set_xlabel("U (V)")
    ax2.set_ylabel("I (A)")
    ax2.grid(True)
    ax2.legend()

    # 3. I2 - I1
    ax3 = fig.add_subplot(gs[1, 2])
    I_diff_closed = close_array(I_diff)
    U_smooth = smooth_savgol(close_array(U2), window_length=21, polyorder=5)
    I_smooth = smooth_savgol(I_diff_closed, window_length=21, polyorder=5)
    ax3.plot(U_smooth, I_smooth, label='I2 - I1', color='purple')
    ax3.set_title("I2 - I1 sau căn pha")
    ax3.set_xlabel("U (V)")
    ax3.set_ylabel("ΔI (A)")
    ax3.grid(True)
    ax3.legend()

    # 4. Ảnh trắng đen
    ax4 = fig.add_subplot(gs[1, 3])
    img = plot_to_bw_image(U2_closed, I_diff_closed, 32, 32)
    ax4.imshow(img, cmap='gray')
    ax4.set_title("Ảnh trắng đen từ I2 - I1")
    ax4.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class MLZLOrigin:
    def __init__(self, window_size=6, power_threshold=30,
                 time_threshold=0.2, deriv_threshold=5,
                 long_transient_time=2, sample_rate=20):
        self.window_size = window_size
        self.power_threshold = power_threshold
        self.time_threshold = time_threshold
        self.deriv_threshold = deriv_threshold
        self.long_transient_time = long_transient_time
        self.sample_rate = sample_rate
        
        # bộ đệm tín hiệu trượt
        self.buffer = []
        self.last_event_time = -np.inf  # thời gian sự kiện cuối cùng (s)
        self.time_index = 0  # đếm thời gian

    # -------------------------------
    # Thêm một giá trị công suất mới
    # -------------------------------
    def update(self, power_value):
        self.time_index += 1
        self.buffer.append(power_value)
        
        # nếu chưa đủ dữ liệu => không phát hiện
        if len(self.buffer) < 2 * self.window_size:
            return 0

        # --- Base Algorithm (A1) ---
        before = self.buffer[:self.window_size]
        after = self.buffer[-self.window_size:]
        mean_before = np.mean(before)
        mean_after = np.mean(after)
        delta = mean_after - mean_before
        
        event_flag = 0
        if abs(delta) > self.power_threshold:
            if delta > 0:
                event_flag = 1   # bật
            else:
                event_flag = -1  # tắt

        # --- Time Limit Check (A2) ---
        current_time = self.time_index / self.sample_rate
        if current_time - self.last_event_time < self.time_threshold:
            event_flag = 0  # gộp sự kiện gần
        if event_flag != 0:
            self.last_event_time = current_time

        # --- Derivative Analysis (B) ---
        if event_flag != 0:
            deriv = np.diff(self.buffer)
            if len(deriv) >= 9:
                smooth_deriv = savgol_filter(deriv, window_length=9, polyorder=3)
                # Nếu đạo hàm nhỏ → có thể là nhiễu, bỏ qua
                if np.max(np.abs(smooth_deriv[-self.window_size:])) < self.deriv_threshold:
                    event_flag = 0

        # --- Filtering Analysis (C) ---
        if event_flag != 0:
            if len(self.buffer) >= 9:
                filtered = savgol_filter(self.buffer, window_length=9, polyorder=3)
                mean_bf = np.mean(filtered[:self.window_size])
                mean_af = np.mean(filtered[-self.window_size:])
                if abs(mean_af - mean_bf) < self.power_threshold:
                    event_flag = 0  # bị triệt tiêu bởi lọc, không phải sự kiện thật

        # if event_flag != 0:
        #     fig, axes = plt.subplots(4, 1, figsize=(8, 10))  # 4 hàng, 1 cột

        #     axes[0].plot(self.buffer)
        #     axes[0].set_title("Cửa sổ gốc")

        #     axes[1].plot(deriv)
        #     axes[1].set_title("Đạo hàm")

        #     axes[2].plot(smooth_deriv)
        #     axes[2].set_title("Đạo hàm làm mượt")

        #     axes[3].plot(filtered)
        #     axes[3].set_title("Làm mượt")

        #     plt.tight_layout()  # tự canh khoảng cách giữa các đồ thị
        #     plt.show()
        #--- Reset buffer ---
        # Giữ lại một nửa sau của cửa sổ để phát hiện liên tục (như trượt)
        self.buffer = self.buffer[self.window_size:]
        return event_flag

# Tạo dữ liệu giả lập 60 giây (20Hz)
# np.random.seed(0)
# signal = np.ones(2500) * 100
# signal[200:500] += 50    
# signal[850:1300] -= 30  
# signal[1700:2500] -= 30  
# signal += np.random.normal(0, 3, len(signal))  # nhiễu

# detector = MLZL(window_size=6, power_threshold=20)
# events = []

# for i, p in enumerate(signal):
#     flag = detector.update(p)
#     if flag != 0:
#         events.append((i, flag))

# # Hiển thị kết quả
# plt.figure(figsize=(12,5))
# plt.plot(signal, label='Power signal')
# for i, flag in events:
#     color = 'r' if flag == 1 else 'b'
#     plt.axvline(i, color=color, linestyle='--', alpha=0.7)
# plt.legend()
# plt.title('Hybrid Event Detection (Streaming Version)')
# plt.xlabel('Sample Index')
# plt.ylabel('Power (W)')
# plt.show()

# print("Detected events:", events)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# --- Đọc dữ liệu từ file ---
csv_path = r"ElectricDatas\MyData\data csv 2\sacmt_event_quat.csv"
parts = csv_path.replace("\\", "/").split("/")
csv_path = os.path.join(*parts)

df = pd.read_csv(csv_path)
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values

# --- Cấu hình ---
Fs = 1000                  # Tần số lấy mẫu (Hz)
f_line = 50                # Tần số lưới điện (Hz)
samples_per_cycle = Fs // f_line  # ~16-17 mẫu mỗi chu kỳ

window_len = 300           # Cửa sổ trượt DBSCAN (chu kỳ)
eps = 0.5                  # DBSCAN epsilon
min_pts = 10               # DBSCAN min_samples
P_threshold = 25           # Ngưỡng chênh lệch công suất để chấp nhận sự kiện
T_min_gap = 10             # Khoảng cách tối thiểu giữa 2 sự kiện (chu kỳ)

# --- Hàm tính RMS theo từng chu kỳ ---
def calc_rms(signal, N):
    return np.array([
        np.sqrt(np.mean(signal[i:i+N]**2))
        for i in range(0, len(signal) - N, N)
    ])

# --- Tính RMS dòng điện và công suất ---
Irms = calc_rms(I_raw, samples_per_cycle)
Prms = calc_rms(Power, samples_per_cycle)

# --- Phát hiện sự kiện sử dụng DBSCAN theo cửa sổ trượt ---
detected_events = []
last_event_idx = -T_min_gap

for t in range(0, len(Irms) - window_len):
    # Dữ liệu trong cửa sổ
    X_window = np.stack([Irms[t:t+window_len], Prms[t:t+window_len]], axis=1)
    X_scaled = StandardScaler().fit_transform(X_window)

    # Áp dụng DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(X_scaled)
    labels = db.labels_

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    # Chỉ xử lý khi có đúng 2 cụm (không tính nhiễu)
    if len(unique_labels) == 2:
        # Tìm chỉ số biên chuyển cụm
        change_idx = np.where(np.diff(labels) != 0)[0]
        if len(change_idx) > 0:
            event_idx = t + change_idx[0]
            P1 = Prms[event_idx]
            P2 = Prms[event_idx + 1]
            delta_P = abs(P2 - P1)

            # Nếu chênh lệch đủ lớn và cách sự kiện trước đó đủ xa
            if delta_P > P_threshold and (event_idx - last_event_idx) > T_min_gap:
                detected_events.append(event_idx)
                last_event_idx = event_idx

# --- Vẽ biểu đồ ---
plt.figure(figsize=(14, 6))
plt.plot(Prms, label="Power (RMS)", color='black')
plt.scatter(detected_events, Prms[detected_events], color='red', label="Detected Events", zorder=5)
plt.title("Phát hiện sự kiện ON/OFF bằng DBSCAN (cửa sổ trượt)")
plt.xlabel("Chu kỳ (RMS)")
plt.ylabel("Công suất (W)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Xuất sự kiện ---
print("Số lượng sự kiện phát hiện:", len(detected_events))
print("Các chu kỳ xảy ra sự kiện:", detected_events)

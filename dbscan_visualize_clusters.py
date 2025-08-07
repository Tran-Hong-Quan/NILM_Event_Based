import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# --- Đọc dữ liệu ---
csv_path = r"ElectricDatas\MyData\data csv\quat_mayep_maysay_tulanh_event_sacmt.csv"
parts = csv_path.replace("\\", "/").split("/")
csv_path = os.path.join(*parts)

df = pd.read_csv(csv_path)
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values

# --- Thông số ---
Fs = 1000
f_line = 60
samples_per_cycle = Fs // f_line

# --- Hàm RMS theo chu kỳ ---
def calc_rms(signal, N):
    return np.array([
        np.sqrt(np.mean(signal[i:i+N]**2))
        for i in range(0, len(signal) - N, N)
    ])

Irms = calc_rms(I_raw, samples_per_cycle)
Prms = calc_rms(Power, samples_per_cycle)

# --- Vùng kiểm tra ---
start = 3800  # chu kỳ RMS bắt đầu
end = 4400    # chu kỳ RMS kết thúc

Irms_seg = Irms[start:end]
Prms_seg = Prms[start:end]
X_seg = np.stack([Irms_seg, Prms_seg], axis=1)
X_scaled = StandardScaler().fit_transform(X_seg)

# --- DBSCAN ---
eps = 0.5
min_pts = 10
db = DBSCAN(eps=eps, min_samples=min_pts).fit(X_scaled)
labels = db.labels_

# === Biểu đồ 1: Toàn bộ công suất ===
plt.figure(figsize=(14, 4))
plt.plot(Prms, color='gray', linewidth=1, label='Tín hiệu công suất gốc')
plt.axvspan(start, end, color='orange', alpha=0.3, label='Vùng đang kiểm tra')
plt.title("Toàn bộ tín hiệu công suất và vùng được kiểm tra")
plt.ylabel("Công suất (W)")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()

# === Biểu đồ 3: Phân cụm trong không gian đặc trưng chuẩn hóa ===
plt.figure(figsize=(8, 6))

# Vẽ từng cụm
unique_labels = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    idx = labels == label
    if label == -1:
        plt.scatter(X_scaled[idx, 0], X_scaled[idx, 1], c='red', s=30, label='Noise', edgecolors='k')
    else:
        plt.scatter(X_scaled[idx, 0], X_scaled[idx, 1], c=[color], s=30, label=f'Cluster {label}', edgecolors='k')

# Định dạng biểu đồ
plt.title("Phân cụm DBSCAN trong không gian đặc trưng chuẩn hóa (Irms - Prms)")
plt.xlabel("Irms (chuẩn hóa)")
plt.ylabel("Prms (chuẩn hóa)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


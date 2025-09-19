import subprocess
import os
from glob import glob

# ==== Thư mục CSV ====
folder_path = os.path.join("ElectricDatas", "MyData", "NewNew", "unprocessed_data")

# ==== Danh sách lưu kết quả ====
true_labels_list = []
pred_labels_list = []
sum_evt = 0

# ==== Vòng lặp test từng file CSV ====
for csv_file in glob(os.path.join(folder_path, "*.csv")):
    file_name = os.path.basename(csv_file)
    print(f"\n📂 Đang xử lý file: {file_name}")
    
    # Chạy FullSystem.py
    result = subprocess.run(
        ["python", "FullSystem.py", csv_file],
        capture_output=True,
        text=True
    )

    # Lấy LABEL dự đoán
    evt_count = 0
    for line in result.stdout.splitlines():
        if line.startswith("Event_count="):
            evt_count = int(line.replace("Event_count=", "").strip())
            break

    print(f"Event {evt_count}/5")
    sum_evt += evt_count

print("Sum events =", sum_evt)

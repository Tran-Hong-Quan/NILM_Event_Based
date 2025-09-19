import subprocess
import os
from glob import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_device_name(csv_path: str) -> str:
    # Lấy tên file
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    tokens = filename.split("_")
    
    # Tìm "event" và lấy phần sau cùng (chính là tên thiết bị)
    if "event" in tokens:
        return tokens[-1]  # ví dụ: maysay, quat, tulanh
    return None

# ==== Thư mục CSV ====
folder_path = os.path.join("ElectricDatas", "MyNewData")

# ==== Danh sách lưu kết quả ====
true_labels_list = []
pred_labels_list = []

# ==== Vòng lặp test từng file CSV ====
for csv_file in glob(os.path.join(folder_path, "*.csv")):
    file_name = os.path.basename(csv_file)
    true_label = get_device_name(file_name)
    #print(true_label)
    if(true_label == None) :
        continue
    print(f"\n📂 Đang xử lý file: {file_name}")
    
    # Chạy FullSystem.py
    result = subprocess.run(
        ["python", "FullSystem.py", csv_file],
        capture_output=True,
        text=True
    )

    # Lấy LABEL dự đoán
    predicted_label = None
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_LABEL="):
            predicted_label = line.replace("RESULT_LABEL=", "").strip()
            break
        # Debug nếu cần xem output từng dòng:
        # print("line = " + line)

    if predicted_label is None:
        print(f"❌ Không tìm thấy LABEL trong output cho {file_name}")
        continue

    # Lưu kết quả để đánh giá
    true_labels_list.append(true_label)
    pred_labels_list.append(predicted_label)

    # In kết quả từng file
    match_status = "✅ ĐÚNG" if predicted_label == true_label else "❌ SAI"
    print(f"📌 Label thật: {true_label} | Dự đoán: {predicted_label} --> {match_status}")

# ==== ĐÁNH GIÁ TOÀN BỘ ====
if true_labels_list:
    acc = accuracy_score(true_labels_list, pred_labels_list)
    print("\n===== ĐÁNH GIÁ MÔ HÌNH =====")
    print(f"🎯 Accuracy: {acc*100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(true_labels_list, pred_labels_list))

    # Vẽ Confusion Matrix
    labels_unique = sorted(list(set(true_labels_list + pred_labels_list)))
    cm = confusion_matrix(true_labels_list, pred_labels_list, labels=labels_unique)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_unique, yticklabels=labels_unique)
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title(f"Confusion Matrix - Accuracy: {acc*100:.2f}%")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Không có kết quả nào để đánh giá.")

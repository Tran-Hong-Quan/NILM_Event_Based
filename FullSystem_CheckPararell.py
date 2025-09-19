import subprocess
import os
from glob import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_device_name(csv_path: str) -> str:
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    tokens = filename.split("_")
    if "event" in tokens:
        return tokens[-1]
    return None


def run_one_file(csv_file: str):
    """Hàm xử lý 1 file CSV"""
    file_name = os.path.basename(csv_file)
    true_label = get_device_name(file_name)
    if true_label is None:
        return None, None, file_name

    result = subprocess.run(
        ["python", "FullSystem.py", csv_file],
        capture_output=True,
        text=True
    )

    predicted_label = None
    #print(result.stdout.splitlines())
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_LABEL="):
            predicted_label = line.replace("RESULT_LABEL=", "").strip()
            break

    return true_label, predicted_label, file_name


if __name__ == "__main__":  # ⚠️ BẮT BUỘC để tránh lỗi RuntimeError khi chạy song song trên Windows
    folder_path = os.path.join("ElectricDatas", "MyNewData")

    true_labels_list = []
    pred_labels_list = []

    csv_files = glob(os.path.join(folder_path, "*.csv"))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_one_file, csv) for csv in csv_files]

        for future in as_completed(futures):
            true_label, predicted_label, file_name = future.result()

            if true_label is None:
                continue

            if predicted_label is None:
                print(f"❌ Không tìm thấy LABEL trong output cho {file_name}")
                continue

            true_labels_list.append(true_label)
            pred_labels_list.append(predicted_label)

            match_status = "✅ ĐÚNG" if predicted_label == true_label else "❌ SAI"
            print(f"\n📂 {file_name}")
            print(f"📌 Label thật: {true_label} | Dự đoán: {predicted_label} --> {match_status}")

    # ==== ĐÁNH GIÁ TOÀN BỘ ====
    if true_labels_list:
        acc = accuracy_score(true_labels_list, pred_labels_list)
        print("\n===== ĐÁNH GIÁ MÔ HÌNH =====")
        print(f"🎯 Accuracy: {acc*100:.2f}%")
        print("\n📊 Classification Report:")
        print(classification_report(true_labels_list, pred_labels_list))

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

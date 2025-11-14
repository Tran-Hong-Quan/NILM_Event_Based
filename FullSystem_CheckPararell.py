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


def run_one_file(csv_file: str,pythonFile: str):
    """Hàm xử lý 1 file CSV"""
    file_name = os.path.basename(csv_file)
    true_label = get_device_name(file_name)
    if true_label is None:
        return None, None, file_name

    result = subprocess.run(
        ["python", pythonFile, csv_file],
        capture_output=True,
        text=True
    )

    predicted_label = []
    event_count = 0
    print(result.stderr)
    for line in result.stdout.splitlines():
        #print(line)
        if line.startswith("RESULT_LABEL="):
            predicted_label.append(line.replace("RESULT_LABEL=", "").strip())
        if line.startswith("EVENT_COUNT="):
            event_count = int(line.replace("EVENT_COUNT=", "").strip())

    return true_label, predicted_label, file_name, event_count


if __name__ == "__main__":  # ⚠️ BẮT BUỘC để tránh lỗi RuntimeError khi chạy song song trên Windows
    folder_path = os.path.join("ElectricDatas", "MyNewData")

    true_labels_list = []
    pred_labels_list = []
    event_count = 0
    missedEventCount = 0
    fakeEventCount = 0
    evaluateOnlyML = False
    pythonFile = "FullSystemWAMMA.py"

    csv_files = glob(os.path.join(folder_path, "*.csv"))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_one_file, csv, pythonFile) for csv in csv_files]

        for future in as_completed(futures):
            true_label, predicted_labels, file_name, file_event_count = future.result()

            if true_label is None:
                continue

            if predicted_labels is None:
                print(f"❌ Không tìm thấy LABEL trong output cho {file_name}")
                continue
            
            isMatch = False
            predicted = ""
            if len(predicted_labels) > 0:
                for p in predicted_labels:
                    if p in true_label:
                        true_labels_list.append(p)
                        pred_labels_list.append(p)
                        
                        predicted = p
                        predicted_labels.remove(p)
                        isMatch = True
                        break
                if not isMatch:
                    true_labels_list.append(true_label)
                    pred_labels_list.append(predicted_labels[0])
                    
                    predicted = predicted_labels[0]
                    predicted_labels.remove(predicted_labels[0])
                if not evaluateOnlyML:
                    for p in predicted_labels:
                            true_labels_list.append("null")
                            pred_labels_list.append(p)
            elif not evaluateOnlyML:
                true_labels_list.append(true_label)
                pred_labels_list.append("null")
            
            match_status = "✅ ĐÚNG" if isMatch else "❌ SAI"
            
            event_count += file_event_count   
                
            print(f"\n📂 {file_name}")
            if file_event_count > 1:
                print(f"Có {file_event_count}")
                fakeEventCount += file_event_count - 1
            elif file_event_count == 0:
                print("Không thấy event")
                missedEventCount += 1
            print(f"📌 Label thật: {true_label} | Dự đoán: {predicted} --> {match_status}")

    # ==== ĐÁNH GIÁ TOÀN BỘ ====
    if true_labels_list:
        acc = accuracy_score(true_labels_list, pred_labels_list)
        print("\n===== ĐÁNH GIÁ MÔ HÌNH =====")
        print(f"🎯 Accuracy: {acc*100:.2f}%")
        print("\n📊 Classification Report:")
        print(classification_report(true_labels_list, pred_labels_list))
        print("Số lượng event / số lượng event thực tế = " + str(event_count) + " / " +str(len(csv_files)))
        print(f"Số lượng event giả: {fakeEventCount}")
        print(f"Số lượng event bỏ lỡ: {missedEventCount}")

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

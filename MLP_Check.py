import pandas as pd
import numpy as np
import os
import glob
from Utilis.NILM_Utilis import (CycleInterpolator, align_phase, calc_prms, 
                                smooth_savgol, flip_ui_image, plot_to_bw_image_with_gaussian_dots)
from DrawUIImage import plt_ui_full_onefig
from MLP_Predict import MLP_Predict
from PIL import Image
from Template_Matcher import TemplateMatcher  

# -----------------------------Tham số hệ thống------------------------
SAMPLING_RATE = 1000        
FREQUENCY = 50              
IMAGE_CYCLE_DURATION = 2    
INTERP_FACTOR = 10          
SAMPLES_PER_CYCLE = SAMPLING_RATE // FREQUENCY
IMAGE_CYCLES = int(IMAGE_CYCLE_DURATION * FREQUENCY)
SAMPLE_PER_IMAGE = SAMPLES_PER_CYCLE * IMAGE_CYCLES

# -----------------------------Khởi tạo MLP----------------------------
clf = MLP_Predict(
    model_path="MLP.pth",
    label_encoder_path="MLP_label_encoder.pkl"
)
matcher = TemplateMatcher("")
CONFIDENCE_THRESHOLD = 0.8

# -----------------------------Hàm xử lý-------------------------------
def cal_img(I_raw, U_raw, Power, start1, start2):
    i1 = I_raw[start1 : start1 + SAMPLE_PER_IMAGE]
    u1 = U_raw[start1 : start1 + SAMPLE_PER_IMAGE]
    i2 = I_raw[start2 : start2 + SAMPLE_PER_IMAGE]
    u2 = U_raw[start2 : start2 + SAMPLE_PER_IMAGE]

    if len(i1) < SAMPLE_PER_IMAGE or len(i2) < SAMPLE_PER_IMAGE:
        return "null"

    delta_p_mean = abs(calc_prms(i2,u2) - calc_prms(i1,u1))
    if delta_p_mean < 15:
        return "null"

    LAST_CYCLE = CycleInterpolator(SAMPLES_PER_CYCLE, INTERP_FACTOR)
    LAST_CYCLE.update_batch(i1, u1)
    CURRENT_CYCLE = CycleInterpolator(SAMPLES_PER_CYCLE, INTERP_FACTOR)
    CURRENT_CYCLE.update_batch(i2, u2)

    U_LAST, I_LAST = LAST_CYCLE.get_average()
    U_CUR, I_CUR = CURRENT_CYCLE.get_average()

    U_LAST_ALIGNED, best_shift = align_phase(U_CUR, U_LAST)
    I_LAST_ALIGNED = np.roll(I_LAST, -int(best_shift))

    I_RES = smooth_savgol(I_CUR - I_LAST_ALIGNED)
    U_RES = smooth_savgol(U_CUR)

    img_np = plot_to_bw_image_with_gaussian_dots(U_RES, I_RES, 32, 32, 2, 0.3)
    img_np = flip_ui_image(img_np)
    image = Image.fromarray(img_np, mode='L') 

    label, confidence = clf.predict(image_input=img_np, p_mean=delta_p_mean)
    label = matcher.match(label, delta_p_mean)

    if confidence < CONFIDENCE_THRESHOLD:
        label = "null"

    return label

# -----------------------------Chạy toàn bộ file------------------------
folder_path = r"ElectricDatas\MyData\New\data csv"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

correct = 0
total = 0

for csv_file in csv_files:
    file_name = os.path.basename(csv_file)
    if "event_" in file_name:
        true_label = file_name.split("event_")[1].replace(".csv", "")
    else:
        continue  # bỏ qua file không đúng định dạng

    df = pd.read_csv(csv_file)
    Power = df["Power"].values
    I_raw = df["In"].values
    U_raw = df["Un"].values

    # test nhanh, có thể random start
    start1 = 1000
    start2 = 5000
    pred_label = cal_img(I_raw, U_raw, Power, start1, start2)

    print(f"File: {file_name}, True: {true_label}, Pred: {pred_label}")

    if pred_label == true_label:s
        correct += 1
    total += 1

if total > 0:
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}")
else:
    print("⚠️ Không tìm thấy file CSV hợp lệ (có chứa 'event_') trong thư mục.")

import pandas as pd
import numpy as np
import os
from Utilis.NILM_Utilis import (CycleInterpolator, align_phase, calc_prms, 
                                smooth_savgol, flip_ui_image, plot_to_bw_image_with_gaussian_dots)
from DrawUIImage import plt_event_window,plt_ui_full_onefig
from Utilis.EventDectection.MLZLOrigin import MLZLOrigin
from MLP_Predict import MLP_Predict
from PIL import Image
from Template_Matcher import TemplateMatcher  
import numpy as np
import sys
import config

# --- Cấu hình Test ---
isChecking = False
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = r"ElectricDatas\MyNewData\data_9_quat_tulanh_event_on_maysay.csv"
    isChecking = True
parts = csv_path.replace("\\", "/").split("/")
csv_path = os.path.join(*parts)
df = pd.read_csv(csv_path)
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values
data_len = len(I_raw)

#Lấy dữ liệu event
try:
    filename = os.path.basename(csv_path)
    filename_no_ext = os.path.splitext(filename)[0]  # bỏ .csv
    parts = filename_no_ext.split("_")
    event_index = parts.index("event")
    state = parts[event_index + 1]
    device = parts[event_index + 2]
except (ValueError, IndexError):
    state = device = "null"

print("State:", state)
print("Device:", device)

# -----------------------------Tham số hệ thống------------------------
SAMPLING_RATE = 1000        # Tần lấy mẫu bộ đo
FREQUENCY = 50              # Tần số mạng điện 
# -----------Tham số trích xuất ảnh------------------------------------
IMAGE_CYCLE_DURATION = 2    # Thời gian lấy mẫu để tạo ảnh
INTERP_FACTOR = 10           # Nhân tử nội suy
# -----------Tham số bộ phát hiện sự kiện------------------------------
EVENT_SAMPLING_RATE = 100   # Tần số lấy mẫu cho bộ phát hiện sự kiện
WINDOW_SEC = 6        
EVENT_P_THRE = 30          
EVENT_TIME_LIMIT_COMMMON = 6
DERIV_THRESHOLD = 10
LONG_TRANS_LIMIT = 2

#------------Khởi tạo các tham số ngoài---------------------------------
SAMPLES_PER_CYCLE = SAMPLING_RATE // FREQUENCY      # Số điểm ở mỗi vòng
IMAGE_CYCLES = int(IMAGE_CYCLE_DURATION * FREQUENCY)     # Số vòng để tạo ảnh
SAMPLE_PER_IMAGE = SAMPLES_PER_CYCLE * IMAGE_CYCLES
P_EVENT_BUFFER = []                                 # Mảng lưu giá trị tính trung bình cho bộ phát hiện sự kiên
P_EVENT_BUFFER_LEN = SAMPLING_RATE // EVENT_SAMPLING_RATE   # Độ dài buffer cho tính trung bình công suất cho phát hiện sự kiến
last_event_time = -1
#---------------Tham số debug/đánh giá-----------------------
evt_count = 0
IsRightLabel = False

# Khởi tạo bộ phát hiện sự kiện
eventDetection = MLZLOrigin(int(WINDOW_SEC * EVENT_SAMPLING_RATE),
                           EVENT_P_THRE,
                           EVENT_TIME_LIMIT_COMMMON,DERIV_THRESHOLD,LONG_TRANS_LIMIT,EVENT_SAMPLING_RATE)

#Khởi tạo mô hình MLP
clf = MLP_Predict(
    model_path="ML_DATA/MLP.pth",
    label_encoder_path="ML_DATA/MLP_label_encoder.pkl"
)

#Khởi tạo hậu xử lý dữ liệu
matcher = TemplateMatcher("")
CONFIDENCE_THRESHOLD = 0

# Hàm tính ảnh I2 - I1 và gọi hàm vẽ
def cal_img(start1, start2):
    #print(SAMPLE_PER_IMAGE)
    i1 = I_raw[start1 : start1 + SAMPLE_PER_IMAGE]
    u1 = U_raw[start1 : start1 + SAMPLE_PER_IMAGE]
    i2 = I_raw[start2 : start2 + SAMPLE_PER_IMAGE]
    u2 = U_raw[start2 : start2 + SAMPLE_PER_IMAGE]
    #print(calc_prms(i2,u2))
    #print(calc_prms(i1,u1))
    delta_p_mean = abs(calc_prms(i2,u2) - calc_prms(i1,u1))
    print("Delta P RMS = "  +str(delta_p_mean))
    if delta_p_mean < 20:
        return

    if len(i1) < SAMPLE_PER_IMAGE or len(i2) < SAMPLE_PER_IMAGE:
        #print("[Warning] Không đủ mẫu để tạo ảnh.")
        return

    LAST_CYCLE = CycleInterpolator(SAMPLES_PER_CYCLE, INTERP_FACTOR)
    LAST_CYCLE.update_batch(i1, u1)
    CURRENT_CYCLE = CycleInterpolator(SAMPLES_PER_CYCLE, INTERP_FACTOR)
    CURRENT_CYCLE.update_batch(i2, u2)

    U_LAST, I_LAST = LAST_CYCLE.get_average()
    U_CUR, I_CUR = CURRENT_CYCLE.get_average()

    _, best_shift = align_phase(U_CUR, U_LAST)
    I_LAST_ALIGNED = np.roll(I_LAST, -int(best_shift))
    I_RES = (I_CUR - I_LAST_ALIGNED)
    U_RES = U_CUR
    U_RES = smooth_savgol(U_RES)
    I_RES = smooth_savgol(I_RES)
    
    img_np = plot_to_bw_image_with_gaussian_dots(U_RES, I_RES, config.IMAGE_SIZE, config.IMAGE_SIZE,config.IMG_DOT_RADIUS,config.IMG_DOT_ALPHA)
    img_np = flip_ui_image(img_np)
    label,confidence  = clf.predict(image_input=img_np, p_mean=delta_p_mean)
    print("MLP label: " + str(label))
    if label == None:
        label = "null"
    # label = matcher.match(label,delta_p_mean)
    if isChecking:
        image = Image.fromarray(img_np, mode='L') 
        plt_ui_full_onefig(SAMPLING_RATE, Power,
                    start1, start1 + SAMPLE_PER_IMAGE,
                    start2, start2 + SAMPLE_PER_IMAGE,
                    U_LAST, I_LAST, U_CUR, I_CUR, I_RES,image,delta_p_mean,label,confidence)
    if confidence < CONFIDENCE_THRESHOLD:
        label = "null"
    return label
# -------------------- Vòng lặp chính --------------------
#plt_event_window(Power,1000,0,10,0)
for idx in range(data_len):
    i = I_raw[idx]
    u = U_raw[idx]
    p = abs(i * u)
    P_EVENT_BUFFER.append(p)

    if len(P_EVENT_BUFFER) == P_EVENT_BUFFER_LEN:
        event = eventDetection.update(np.mean(P_EVENT_BUFFER))
        winDuration = WINDOW_SEC
        P_EVENT_BUFFER = []

        if event != 0:
            FWD_SEC = 0        
            BWD_SEC = winDuration + 6             
            base = idx
            step = SAMPLE_PER_IMAGE
            start1 = base - int(BWD_SEC * SAMPLING_RATE)
            start2 = base + int(FWD_SEC * SAMPLING_RATE)

            if start2 > 0 and start2 + step < data_len and start1 > 0:
                label = cal_img(start1, start2)
                P_Mean = calc_prms(U_raw[start2 : start2 + SAMPLE_PER_IMAGE],I_raw[start2 : start2 + SAMPLE_PER_IMAGE])
                #print("P_Mean = " + str(P_Mean))

                if label == "null" or label is None:
                    continue

                # --- Kiểm tra thời gian giới hạn ---
                delta_time = (idx - last_event_time) / SAMPLING_RATE
                accept_event = True

                if last_event_time >= 0:  # phải có event trước mới so sánh được
                    if delta_time <= EVENT_TIME_LIMIT_COMMMON:
                        accept_event = False
                        print("fake event")
                        

                # Nếu event thật
                if accept_event:
                    last_event_time = idx
                    evt_count += 1
                    if label != "null":  
                        print(f"RESULT_LABEL={label}")
                        if label == device:
                            IsRightLabel = True
                        
print("EVENT_COUNT="+str(evt_count))
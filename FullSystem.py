import pandas as pd
import numpy as np
import os
from Utilis.NILM_Utilis import (CycleInterpolator, align_phase, calc_prms, 
                                smooth_savgol, flip_ui_image, plot_to_bw_image_with_gaussian_dots)
from DrawUIImage import plt_event_window,plt_ui_full_onefig
from Utilis.EventDectection.QUAN import QuanDetector
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
WAMMA_WINDOW_SEC = 4        # Cửa sổ wamma
WAMMA_EDGE_SEC = 1        # Biên wamma
LOW_DEC_HZ = 1              # Tần số bộ phát hiện sự kiện tần số thấp
LOW_DEC_WINDOW_SEC = 6      # Cửa số bộ tần số thập
EVENT_TIME_LIMIT_DIF = 11   # Giới hạn thời gian 2 Event Khác nhau
EVENT_TIME_LIMIT_SAM = 5    # Giới hạn thời gian 2 Event Giống nhau
WAMMA_P_THRE = 30           # P giới hạn phát hiện sự kiện cho cửa sổ WAMMA
WAMMA_R_THRE = 1            # R giới hạn phát hiện sự kiện cho WAMMA, càng bé càng nhạy với nhiễu
LOW_DEC_THRE = 20           # P giới hạn phát hiện sự kiện cho cửa sổ phát hiện sự kiện tần số thấp
KALMAN_Q = 0.01             # Q CỦA BỘ LỌC KALMAN
KALMAN_R = 100              # R CỦA BỘ LỌC KALMAN

#------------Khởi tạo các tham số ngoài---------------------------------
SAMPLES_PER_CYCLE = SAMPLING_RATE // FREQUENCY      # Số điểm ở mỗi vòng
IMAGE_CYCLES = int(IMAGE_CYCLE_DURATION * FREQUENCY)     # Số vòng để tạo ảnh
SAMPLE_PER_IMAGE = SAMPLES_PER_CYCLE * IMAGE_CYCLES
P_EVENT_BUFFER = []                                 # Mảng lưu giá trị tính trung bình cho bộ phát hiện sự kiên
P_EVENT_BUFFER_LEN = SAMPLING_RATE // EVENT_SAMPLING_RATE   # Độ dài buffer cho tính trung bình công suất cho phát hiện sự kiến


#--------------- Tham số loại event giả-------------------
EVENT_TIME_LIMIT_COMMMON = 6
EVENT_TYPE_LIMITS = {
    0: 6,   # WAMMA
    1: 14    # LowDec
}
EVENT_WAMMA_2_LOWDEC_LIMITS = 18 # Nếu event trước là wamma thì limit thời gian lowdec
EVENT_LOWDEC_2_WAMMA_LIMITS = 12 # Nếu event trước là lowdec thì limit thời gian wamma
last_event_time = -1
last_event_type = -1
last_label = "null"
last_event_P_Mean = 0
evt_count = 0

#---------------Tham số debug/đánh giá-----------------------
LABEL = "null"
IsRightLabel = False

# Khởi tạo bộ phát hiện sự kiện
quan = QuanDetector(event_sampling_rate = EVENT_SAMPLING_RATE,
                    wamma_window_sec = WAMMA_WINDOW_SEC,
                    wamma_edge_sec= WAMMA_EDGE_SEC,
                    low_dec_sampling_rate = LOW_DEC_HZ,
                    low_dec_window_sec = LOW_DEC_WINDOW_SEC,
                    wamma_p_threshold = WAMMA_P_THRE,
                    wamma_r_threshold = WAMMA_R_THRE,
                    low_dec_threshold =LOW_DEC_THRE,
                    event_time_limit_dif=EVENT_TIME_LIMIT_DIF,
                    event_time_limit_sam=EVENT_TIME_LIMIT_SAM,
                    init_power= 0, 
                    kalman_Q = KALMAN_Q,kalman_R = KALMAN_R)

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
        event, winDuration, eventType = quan.update(np.mean(P_EVENT_BUFFER))
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
                    # 1. Cùng loại + cùng label
                    if last_event_type == eventType and label == last_label:
                        limit_time = EVENT_TYPE_LIMITS.get(eventType, 10)
                        if delta_time < limit_time:
                            #print(f"[Fake Event] quá gần (Δt={delta_time:.2f}s), cùng loại & cùng label {label}")
                            accept_event = False
                            print("fake event")

                    # 2. Khác loại + cùng label
                    elif last_event_type != eventType and label == last_label:
                        if eventType == 0:
                            limit_time = EVENT_LOWDEC_2_WAMMA_LIMITS
                        else:
                            limit_time = EVENT_WAMMA_2_LOWDEC_LIMITS
                        if delta_time < limit_time and abs(last_event_P_Mean - P_Mean) / P_Mean < .2:
                            #print(f"[Fake Event] quá gần (Δt={delta_time:.2f}s), khác loại nhưng cùng label {label}")
                            accept_event = False
                            print("fake event")

                    # 3. Label khác nhau, nhưng P_Mean gần giống
                    elif label != last_label and abs(last_event_P_Mean - P_Mean) / P_Mean < .2:
                        if eventType == last_event_type:
                            limit_time = EVENT_TYPE_LIMITS.get(eventType, 10)
                        else:
                            if eventType == 0:
                                limit_time = EVENT_LOWDEC_2_WAMMA_LIMITS
                            else:
                                limit_time = EVENT_WAMMA_2_LOWDEC_LIMITS
                        print(f"Limit time {limit_time}, delta time {delta_time}")
                        if delta_time < limit_time:
                            #print(f"[Fake Event] Δt={delta_time:.2f}s, label khác ({last_label}→{label}) nhưng P_Mean gần giống")
                            accept_event = False
                            print("fake event")
                    # 4. Event quá gần nhau
                    elif delta_time <= EVENT_TIME_LIMIT_COMMMON:
                        accept_event = False
                        print("fake event")
                        

                # Nếu event thật
                if accept_event:
                    last_event_time = idx
                    last_event_type = eventType
                    last_label = label
                    last_event_P_Mean = P_Mean
                    #print(f"[Real Event] idx={idx}, label={label}, type={eventType}, Δt={(0 if last_event_time<0 else delta_time):.2f}s")
                    evt_count += 1
                    if label != "null":  
                        LABEL = label
                        if LABEL == device:
                            IsRightLabel = True
                            print(f"RESULT_LABEL={LABEL}")
                        
print("EVENT_COUNT="+str(evt_count))
if device != LABEL and IsRightLabel == False:
    print(f"RESULT_LABEL={LABEL}")
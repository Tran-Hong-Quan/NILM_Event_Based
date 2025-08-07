import pandas as pd
import numpy as np
import os
from Utilis.NILM_Utilis import (CycleInterpolator, CircularBuffer, align_phase, close_curve, 
                                close_array, calc_prms, plot_to_bw_image, 
                                smooth_savgol, is_right_side_greater, plot_to_bw_image_with_gaussian_dots)
from DrawUIImage import plt_ui_full, plt_event_window,plt_ui_full_onefig
from Utilis.EventDectection.QUAN import QuanDetector
from MLP_Predict import MLP_Predict
from PIL import Image

# --- Cấu hình Test ---
csv_path = r"ElectricDatas\MyData\data csv 2\sacmt_event_tulanh.csv"
parts = csv_path.replace("\\", "/").split("/")
csv_path = os.path.join(*parts)
df = pd.read_csv(csv_path)
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values
data_len = len(I_raw)

# -----------------------------Tham số hệ thống------------------------
SAMPLING_RATE = 1000        # Tần lấy mẫu bộ đo
FREQUENCY = 50              # Tần số mạng điện 
BUFFER_DURATION = 60        # Độ dài tính bằng thời gian cho bộ đệm lưu I và U
# -----------Tham số trích xuất ảnh------------------------------------
IMAGE_CYCLE_DURATION = 2    # Thời gian lấy mẫu để tạo ảnh
INTERP_FACTOR = 10           # Nhân tử nội suy
# -----------Tham số bộ phát hiện sự kiện------------------------------
EVENT_SAMPLING_RATE = 100   # Tần số lấy mẫu cho bộ phát hiện sự kiện
WAMMA_WINDOW_SEC = 4        # Cửa sổ wamma
WAMMA_EDGE_SEC = 1        # Biên wamma
LOW_DEC_HZ = 1              # Tần số bộ phát hiện sự kiện tần số thấp
LOW_DEC_WINDOW_SEC = 6      # Cửa số bộ tần số thập
EVENT_TIME_LIMIT_DIF = 8   # Giới hạn thời gian 2 Event Khác nhau
EVENT_TIME_LIMIT_SAM = 6    # Giới hạn thời gian 2 Event Giống nhau
WAMMA_P_THRE = 30           # P giới hạn phát hiện sự kiện cho cửa sổ WAMMA
WAMMA_R_THRE = 2            # R giới hạn phát hiện sự kiện cho WAMMA, càng bé càng nhạy với nhiễu
LOW_DEC_THRE = 10           # P giới hạn phát hiện sự kiện cho cửa sổ phát hiện sự kiện tần số thấp
KALMAN_Q = 0.01             # Q CỦA BỘ LỌC KALMAN
KALMAN_R = 1000              # R CỦA BỘ LỌC KALMAN

#------------Khởi tạo các tham số ngoài---------------------------------
SAMPLES_PER_CYCLE = SAMPLING_RATE // FREQUENCY      # Số điểm ở mỗi vòng
IMAGE_CYCLES = int(IMAGE_CYCLE_DURATION * FREQUENCY)     # Số vòng để tạo ảnh
SAMPLE_PER_IMAGE = SAMPLES_PER_CYCLE * IMAGE_CYCLES
BUFFER_LEN = BUFFER_DURATION * SAMPLING_RATE        # Độ dài buffer lưu dữ liệu U, I
I_BUFFER = CircularBuffer(BUFFER_LEN)               # Mảng lưu trữ I của hệ thống đo
U_BUFFER = CircularBuffer(BUFFER_LEN)               # Mảng lưu trư U của hệ thống đo
P_EVENT_BUFFER = []                                 # Mảng lưu giá trị tính trung bình cho bộ phát hiện sự kiên
P_EVENT_BUFFER_LEN = SAMPLING_RATE // EVENT_SAMPLING_RATE   # Độ dài buffer cho tính trung bình công suất cho phát hiện sự kiến
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
state = 0   # Trạng thái hệ thống, -1 là đang khởi tạo, 0 là đang tìm event, 1 là đang thu thập dữ liệu cho nhận diện
currentCycleCount = 0   # Số vòng đã thu thập được cho ảnh

#Khởi tạo mô hình MLP
clf = MLP_Predict(
    model_path="MLP.pth",
    label_encoder_path="label_encoder.pkl"
)

# Hàm tính ảnh I2 - I1 và gọi hàm vẽ
def cal_img(start1, start2, idx):
    #print(SAMPLE_PER_IMAGE)
    i1 = I_raw[start1 : start1 + SAMPLE_PER_IMAGE]
    u1 = U_raw[start1 : start1 + SAMPLE_PER_IMAGE]
    i2 = I_raw[start2 : start2 + SAMPLE_PER_IMAGE]
    u2 = U_raw[start2 : start2 + SAMPLE_PER_IMAGE]

    delta_p_mean = abs(calc_prms(i2,u2) - calc_prms(i1,u1))
    print("Delta P RMS = "  +str(delta_p_mean))
    if(delta_p_mean < 15) :
        return

    if len(i1) < SAMPLE_PER_IMAGE or len(i2) < SAMPLE_PER_IMAGE:
        print("[Warning] Không đủ mẫu để tạo ảnh.")
        return

    LAST_CYCLE = CycleInterpolator(SAMPLES_PER_CYCLE, INTERP_FACTOR)
    LAST_CYCLE.update_batch(i1, u1)
    CURRENT_CYCLE = CycleInterpolator(SAMPLES_PER_CYCLE, INTERP_FACTOR)
    CURRENT_CYCLE.update_batch(i2, u2)

    U_LAST, I_LAST = LAST_CYCLE.get_average()
    U_CUR, I_CUR = CURRENT_CYCLE.get_average()

    U_LAST_ALIGNED, best_shift = align_phase(U_CUR, U_LAST)
    I_LAST_ALIGNED = np.roll(I_LAST, -int(best_shift))
    I_RES = (I_CUR - I_LAST_ALIGNED)
    U_RES = U_CUR
    I_RES *= is_right_side_greater(I_RES, U_RES)
    img_np = plot_to_bw_image_with_gaussian_dots(U_RES, I_RES, 32, 32,1,0.3)
    image = Image.fromarray(img_np, mode='L') 
    label,confidence  = clf.predict(image_input=img_np, p_mean=delta_p_mean)

    plt_ui_full_onefig(SAMPLING_RATE, Power,
                start1, start1 + SAMPLE_PER_IMAGE,
                start2, start2 + SAMPLE_PER_IMAGE,
                U_LAST, I_LAST, U_CUR, I_CUR, I_RES,image,delta_p_mean,label,confidence)
    

# -------------------- Vòng lặp chính --------------------
for idx in range(data_len):
    i = I_raw[idx]
    u = U_raw[idx]
    p = i * u
    P_EVENT_BUFFER.append(p)

    if len(P_EVENT_BUFFER) == P_EVENT_BUFFER_LEN:
        event,winDuration = quan.update(np.mean(P_EVENT_BUFFER))
        P_EVENT_BUFFER = []

        if event != 0:
            winDuration *= 2
            FWD_SEC = 2                  # bao trùm thêm 2 giây sau sự kiện
            BWD_SEC = winDuration + FWD_SEC
            
            base = idx
            step = SAMPLE_PER_IMAGE
            MIN_GAP_SEC = 4
            MIN_GAP_LEN = int(MIN_GAP_SEC * SAMPLING_RATE)

            start_window = base - int(BWD_SEC * SAMPLING_RATE)
            end_window   = base + int(FWD_SEC * SAMPLING_RATE)
            plt_event_window(Power, SAMPLING_RATE, start_window, end_window, idx)

            print(f"[Event] idx={idx}, window: {BWD_SEC}s before, {FWD_SEC}s after, windows size = {winDuration + 2*FWD_SEC}")

            # Duyệt các ảnh sau từ gần end_window
            for after_i in range((end_window - step - start_window) // step):
                start2 = end_window - (after_i + 1) * step + 1

                if start2 < 0 or start2 + step > data_len:
                    continue
                
                # Duyệt các ảnh trước từ start_window
                for before_i in range((end_window - start_window) // step):
                    start1 = start_window + before_i * step

                    if start1 < 0 or start1 + step > data_len:
                        continue
                    
                    if (start2 - (start1 + step)) < MIN_GAP_LEN:
                        break
                    
                    print(f"  -> start2={start2}, start1={start1}, cách={(start2 - start1)/SAMPLING_RATE:.2f}s")
                    cal_img(start1, start2, idx)




            
            
                
    
        
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from Utilis.NILM_Utilis import (CycleInterpolator, CircularBuffer, align_phase, close_curve, 
                                close_array, calc_prms, plot_to_bw_image, 
                                smooth_savgol, is_right_side_greater)
from DrawUIImage import plt_ui_full
from Utilis.EventDectection.QUAN import QuanDetector

# --- Cấu hình Test ---
csv_path = r"ElectricDatas\MyData\data csv\sacmt_maysay_tulanh_event_mayep.csv"
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
IMAGE_CYCLE_DURATION = 1    # Thời gian lấy mẫu để tạo ảnh
INTERP_FACTOR = 6           # Nhân tử nội suy
# -----------Tham số bộ phát hiện sự kiện------------------------------
EVENT_SAMPLING_RATE = 100   # Tần số lấy mẫu cho bộ phát hiện sự kiện
WAMMA_WINDOW_SEC = 40       # Cửa sổ wamma
WAMMA_EDGE_SEC = 1          # Biên wamma
LOW_DEC_HZ = 1              # Tần số bộ phát hiện sự kiện tần số thấp
LOW_DEC_WINDOW_SEC = 6      # Cửa số bộ tần số thập
EVENT_TIME_LIMIT_DIF = 10   # Giới hạn thời gian 2 Event Khác nhau
EVENT_TIME_LIMIT_SAM = 6    # Giới hạn thời gian 2 Event Giống nhau
WAMMA_P_THRE = 30           # P giới hạn phát hiện sự kiện cho cửa sổ WAMMA
WAMMA_R_THRE = 1            # R giới hạn phát hiện sự kiện cho WAMMA, càng bé càng nhạy với nhiễu
LOW_DEC_THRE = 10           # P giới hạn phát hiện sự kiện cho cửa sổ phát hiện sự kiện tần số thấp
KALMAN_Q = 0.01             # Q CỦA BỘ LỌC KALMAN
KALMAN_R = 100              # R CỦA BỘ LỌC KALMAN

#------------Khởi tạo các tham số ngoài---------------------------------
SAMPLES_PER_CYCLE = SAMPLING_RATE // FREQUENCY      # Số điểm ở mỗi vòng
IMAGE_CYCLES = IMAGE_CYCLE_DURATION * FREQUENCY     # Số vòng để tạo ảnh
BUFFER_LEN = BUFFER_DURATION * SAMPLING_RATE        # Độ dài buffer lưu dữ liệu U, I
I_BUFFER = CircularBuffer(BUFFER_LEN)               # Mảng lưu trữ I của hệ thống đo
U_BUFFER = CircularBuffer(BUFFER_LEN)               # Mảng lưu trư U của hệ thống đo
P_EVENT_BUFFER = []                                 # Mảng lưu giá trị tính trung bình cho bộ phát hiện sự kiên
P_EVENT_BUFFER_LEN = SAMPLING_RATE // EVENT_SAMPLING_RATE   # Độ dài buffer cho tính trung bình công suất cho phát hiện sự kiến
# Khởi tạo bộ phát hiện sự kiện
quan = QuanDetector(EVENT_SAMPLING_RATE,WAMMA_WINDOW_SEC,WAMMA_EDGE_SEC,LOW_DEC_HZ,LOW_DEC_WINDOW_SEC,
                    WAMMA_P_THRE,WAMMA_R_THRE,LOW_DEC_THRE,EVENT_TIME_LIMIT_DIF,EVENT_TIME_LIMIT_SAM,
                    0, KALMAN_Q, KALMAN_R)
state = 0   # Trạng thái hệ thống, -1 là đang khởi tạo, 0 là đang tìm event, 1 là đang thu thập dữ liệu cho nhận diện
currentCycleCount = 0   # Số vòng đã thu thập được cho ảnh

def cal_img(start1, start2):
    LAST_CYCLE = CycleInterpolator(SAMPLES_PER_CYCLE,IMAGE_CYCLES) 
    LAST_CYCLE.update_batch(
        I_BUFFER.get_range(start1, start1 + SAMPLES_PER_CYCLE * IMAGE_CYCLES),
        U_BUFFER.get_range(start1, start1 + SAMPLES_PER_CYCLE * IMAGE_CYCLES))
    CURRENT_CYCLE = CycleInterpolator(SAMPLES_PER_CYCLE,IMAGE_CYCLES) 
    CURRENT_CYCLE.update_batch(
        I_BUFFER.get_range(start2, start2 + SAMPLES_PER_CYCLE * IMAGE_CYCLES),
        U_BUFFER.get_range(start2, start2 + SAMPLES_PER_CYCLE * IMAGE_CYCLES))
    U_LAST, I_LAST = LAST_CYCLE.get_average()
    U_CUR, I_CUR = CURRENT_CYCLE.get_average()
    U_LAST_ALIGNED, best_shift = align_phase(U_CUR, U_LAST)
    I_LAST_ALIGNED = np.roll(I_LAST, -best_shift)
    I_RES = (I_CUR - I_LAST_ALIGNED)
    U_RES = U_CUR
    I_RES *= is_right_side_greater(I_RES, U_RES)
    
    plt_ui_full(SAMPLING_RATE,Power,start1,start1 + SAMPLES_PER_CYCLE * IMAGE_CYCLES,
                start2,start2 + SAMPLES_PER_CYCLE * IMAGE_CYCLES,U_LAST, I_LAST, U_CUR, I_CUR, I_RES)
    
    return U_RES, I_RES

# Hàm cập nhật liên tục
for idx in range(data_len):
    i = I_raw[idx]
    u = U_raw[idx]
    I_BUFFER.push(i)
    U_BUFFER.push(u)
    p = i * u

    if len(P_EVENT_BUFFER) == P_EVENT_BUFFER_LEN:
        event, winDuration = quan.update(abs(p))
        P_EVENT_BUFFER = []
        # Phát hiện sự kiện
        if event != 0:
            cal_img(BUFFER_LEN - winDuration * SAMPLING_RATE - SAMPLES_PER_CYCLE * IMAGE_CYCLES, BUFFER_LEN - winDuration * SAMPLING_RATE)
            
            
            
                
    
        
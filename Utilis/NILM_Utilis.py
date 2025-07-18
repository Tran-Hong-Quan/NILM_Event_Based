import numpy as np
from scipy.interpolate import interp1d
import cv2
from scipy.signal import savgol_filter

def plot_to_bw_image(U, I, width, height):
    # Tạo ảnh trắng
    img = np.ones((height, width), dtype=np.uint8) * 255

    # Chuẩn hóa U và I về khoảng [0, width-1] và [0, height-1]
    U_norm = np.interp(U, (min(U), max(U)), (0, width - 1))
    I_norm = np.interp(I, (min(I), max(I)), (height - 1, 0))  # Y ngược trục

    # Vẽ từng đoạn thẳng giữa các điểm
    for i in range(len(U) - 1):
        pt1 = (int(U_norm[i]), int(I_norm[i]))
        pt2 = (int(U_norm[i+1]), int(I_norm[i+1]))
        cv2.line(img, pt1, pt2, color=0, thickness=1)

    return img

def close_curve(x,y):
    return close_array(x),close_array(y)

def close_array(a):
    a_closed = np.append(a,a[0])
    return a_closed

def align_phase(reference, target):
    """
    Dịch pha mảng `target` sao cho giống `reference` nhất,
    bằng cách tìm roll khiến tổng abs sai khác là nhỏ nhất.
    Trả về: target đã được roll, và chỉ số shift đã dùng.
    """
    min_loss = float('inf')
    best_shift = 0
    for shift in range(len(target)):
        rolled = np.roll(target, -shift)
        loss = np.sum(np.abs(rolled - reference))
        if loss < min_loss:
            min_loss = loss
            best_shift = shift
    return np.roll(target, -best_shift), best_shift

def calc_rms(signal: np.ndarray) -> float:
    """
    Tính giá trị hiệu dụng (RMS) của tín hiệu.
    """
    return np.sqrt(np.mean(signal**2))

def calc_prms(U: np.ndarray, I: np.ndarray) -> float:
    return np.mean(U * I)

def smooth_savgol(signal: np.ndarray, window_length: int = 21, polyorder: int = 3) -> np.ndarray:
    """
    Làm mượt tín hiệu bằng bộ lọc Savitzky-Golay.
    window_length: phải là số lẻ.
    polyorder: bậc đa thức (2 hoặc 3 là đủ).
    """
    if window_length % 2 == 0:
        window_length += 1  # phải lẻ
    return savgol_filter(signal, window_length, polyorder)

def is_right_side_greater(I: np.ndarray, V: np.ndarray) -> bool:
    right_sum = np.sum(I[V > 0])
    left_sum = np.sum(I[V < 0])
    
    if right_sum - left_sum > 0:
        return 1
    return -1

class CycleInterpolator:
    def __init__(self, samples_per_cycle, interp_factor=6):
        self.samples_per_cycle = samples_per_cycle
        self.interp_points = samples_per_cycle * interp_factor
        self.clear()

    def clear(self):
        self.I_avg = np.zeros(self.interp_points, dtype=np.float32)
        self.U_avg = np.zeros(self.interp_points, dtype=np.float32)
        self.count = 0
        self.ref_U = None  # dùng cho căn pha

    def update(self, I_cycle, U_cycle):
        # Khép kín vòng
        I_closed, U_closed = close_curve(I_cycle,U_cycle)

        # Tạo trục nội suy
        x_old = np.linspace(0, 1, len(I_closed))
        x_new = np.linspace(0, 1, self.interp_points)

        # Nội suy tuyến tính
        I_interp = interp1d(x_old, I_closed, kind='linear')(x_new).astype(np.float32)
        U_interp = interp1d(x_old, U_closed, kind='linear')(x_new).astype(np.float32)

        # Nếu là vòng đầu tiên thì gán làm ref_U
        if self.ref_U is None:
            self.ref_U = U_interp.copy()

        # Dịch pha U và I cho đồng bộ với ref_U
        U_aligned, shift = align_phase(self.ref_U, U_interp)
        I_aligned = np.roll(I_interp, -shift)

        # Cập nhật trung bình động
        self.count += 1
        alpha = 1.0 / self.count
        self.I_avg = (1 - alpha) * self.I_avg + alpha * I_aligned
        self.U_avg = (1 - alpha) * self.U_avg + alpha * U_aligned
    
    def update_batch(self, I_array, U_array):
        min_len = min(len(I_array), len(U_array))
        max_valid_len = (min_len // self.samples_per_cycle) * self.samples_per_cycle

        if max_valid_len == 0:
            raise ValueError("Không đủ dữ liệu cho một vòng.")

        I_trimmed = I_array[:max_valid_len]
        U_trimmed = U_array[:max_valid_len]
        n_cycles = max_valid_len // self.samples_per_cycle

        for i in range(n_cycles):
            start = i * self.samples_per_cycle
            end = start + self.samples_per_cycle
            self.update(I_trimmed[start:end], U_trimmed[start:end])


    def get_average(self):
        if self.count == 0:
            raise ValueError("Chưa có vòng nào được cập nhật.")
        return self.U_avg, self.I_avg

class CircularBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [0.0] * capacity
        self.head = 0
        self.count = 0

    def push(self, value):
        if self.count < self.capacity:
            self.buffer[(self.head + self.count) % self.capacity] = value
            self.count += 1
        else:
            self.buffer[self.head] = value
            self.head = (self.head + 1) % self.capacity

    def get(self, i):
        if i < 0 or i >= self.count:
            raise IndexError("Index out of range")
        return self.buffer[(self.head + i) % self.capacity]

    def get_range(self, start, length):
        if start < 0 or length < 0 or start + length > self.count:
            raise IndexError("Invalid range")
        return [self.get(i) for i in range(start, start + length)]

    def to_list(self):
        return [self.get(i) for i in range(self.count)]

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        return self.get(i)

    def __str__(self):
        return str(self.to_list())
    


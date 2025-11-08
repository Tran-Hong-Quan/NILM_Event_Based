import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from .iv_image_nam_ver import I_V_image_single_cycle_2
from scipy.interpolate import splprep, splev, interp1d
from scipy.signal import savgol_filter


data_dir = ""

class I_V_data:
    def __init__(self, file, point_per_cycle=20, dir=data_dir):
        self.file = file
        self.folder = data_dir
        self.point_per_cycle = point_per_cycle
        self.df = pd.read_csv(os.path.join(self.folder, file))
        self.In = self.df["In"].values
        self.Un = self.df["Un"].values
        self.Power = self.df["Power"].values

    def plot_data(self, start, end=None):
        if end is None:
            end = len(self.In)
        plt.plot(np.arange(start,end), self.In[start:end], linestyle='-', color='b', linewidth=3)
        plt.title(f'Current data from {start} to {end} of {self.file}')
        plt.ylabel('current (A)')
        plt.xlabel('Index')
        plt.show()

    def Imax_cycle(self, start, end=None):
        if end is None:
            end = len(self.In)
        Imax = []
        for i in range(start, end, self.point_per_cycle):
            Imax.append(self.In[i:i+self.point_per_cycle].max())
        Imax = np.array(Imax)
        plt.plot(np.arange(start, end, self.point_per_cycle), Imax, linestyle='-', color='b', linewidth=3)
        plt.title(f'Biên độ I tức thời from {start} to {end} of {self.file}')
        plt.ylabel('current (A)')
        plt.xlabel('Index')
        plt.show()
    
    def plot_Power(self, start, end=None):
        if end is None:
            end = len(self.Power)
        plt.plot(np.arange(start, end), self.Power[start: end], marker='o', color='r', markersize=5)
        plt.title(f'Công suất from {start} to {end} of {self.file}')
        plt.ylabel('Power (W)')
        plt.xlabel('Index')
        plt.show()
    
    def raw_UI(self, start):
        I_seg = self.In[start: start + self.point_per_cycle*20]
        U_seg = self.Un[start: start + self.point_per_cycle*20]
        return I_seg, U_seg

class MapIV_2:
    def __init__(self, file, point_per_cycle=25, dir=data_dir, N = 16, num_map = 200, plot = False, num_cycle = 20):
        self.file = file
        self.point_per_cycle = point_per_cycle
        self.dir = "ElectricDatas\MyNewData"
        self.num_map = num_map
        self.df = pd.read_csv(os.path.join(dir, file))
        self.In = self.df["In"].values
        self.Un = self.df["Un"].values
        self.Power = self.df["Power"].values
        self.num_cycle = num_cycle
        self.N = N
        self.plot = plot

    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def map_U(self, Un):
        Un_max = Un.max()
        Un_min = Un.min()
        
        num_points = self.num_map  # Tổng số điểm

        phase = np.linspace(np.pi, -np.pi, num_points)

        # Biên độ scale từ sin(x) ∈ [-1, 1] → [Un_min, Un_max]
        amplitude_range = (Un_max - Un_min) / 2
        offset = (Un_max + Un_min) / 2
        u_mapped = offset + amplitude_range * np.sin(phase)

        return u_mapped

    def sapxep_IV(self, In, Un):
        seg_1 = []
        seg_2 = []
        seg_3 = []
        N = len(In)
        for i in range(len(In)):
            Un1, Un2, Un3 = Un[(i-1)%N], Un[i], Un[(i+1)%N]
            if (Un3 > Un2 and Un2 > Un1) or (Un3 < Un2 and Un2 > Un1) or (Un3 > Un2 and Un2 < Un1):
                if Un2 > 0:
                    seg_1.append((In[i], Un[i]))
                if Un2 < 0:
                    seg_3.append((In[i], Un[i]))
            elif Un3 < Un2 and Un2 < Un1:
                seg_2.append((In[i], Un[i]))
        seg_1.sort(key=lambda x: x[1])
        seg_2.sort(key=lambda x: x[1], reverse=True)
        seg_3.sort(key=lambda x: x[1])
        In = np.concatenate([
            np.array(seg_1)[:, 0],
            np.array(seg_2)[:, 0],
            np.array(seg_3)[:, 0]
        ])
        Un = np.concatenate([
            np.array(seg_1)[:, 1], 
            np.array(seg_2)[:, 1], 
            np.array(seg_3)[:, 1]
        ])
        return In, Un

    def sapxep_I(self, In, Un):
        for i in range(len(In)):
            if i+1 == len(In):
                if Un[i] < 0 and Un[0] > 0:
                    idx = i
                    break
            elif Un[i] < 0 and Un[i+1] > 0:
                idx = i
                break

        In = np.concatenate((In[idx:], In[:idx]))
        Un = np.concatenate((Un[idx:], Un[:idx]))
        In = np.append(In, In[0])
        Un = np.append(Un, Un[0])
        return In, Un

    def smooth_vi(self, In, Un, window_size=None, polyorder=2):
        In = np.array(In)
        Un = np.array(Un)
        
        if window_size is None:
            window_size = min(51, len(In) // 10 * 2 + 1)

        current_smoothed = savgol_filter(In, window_size, polyorder)
        voltage_smoothed = savgol_filter(Un, window_size, polyorder)
        
        return np.array(current_smoothed), np.array(voltage_smoothed)

    def map_In_Un(self, In1, In2, Un1, Un2, Un_map):
        In_map = In1 + (In2 - In1) * (Un_map - Un1) / (Un2 - Un1)
        return In_map

    def map_I(self, In, Un, Un_mapped):
        N = len(Un)
        idx = 0
        In_mapped = []
        for i in range(len(Un_mapped)):
            while True:
                if i <= np.argmax(Un_mapped):
                    I1, I2 = In[idx], In[(idx+1)%N]
                    U1, U2 = Un[idx], Un[(idx+1)%N]
                    if Un_mapped[i] >= U1 and Un_mapped[i] <= U2:
                        
                        In_mapped.append(self.map_In_Un(In1=I1, In2=I2, Un1=U1, Un2=U2, Un_map=Un_mapped[i]))
                        break
                    else:
                        idx = (idx + 1) % N
                elif i <= np.argmin(Un_mapped):
                    I1, I2 = In[idx], In[(idx+1)%N]
                    U1, U2 = Un[idx], Un[(idx+1)%N]
                    if Un_mapped[i] <= U1 and Un_mapped[i] >= U2:
                        In_mapped.append(self.map_In_Un(In1=I1, In2=I2, Un1=U1, Un2=U2, Un_map=Un_mapped[i]))
                        break
                    else:
                        idx = (idx + 1) % N
                else:
                    I1, I2 = In[idx], In[(idx+1)%N]
                    U1, U2 = Un[idx], Un[(idx+1)%N]
                    if Un_mapped[i] >= U1 and Un_mapped[i] <= U2:
                        In_mapped.append(self.map_In_Un(In1=I1, In2=I2, Un1=U1, Un2=U2, Un_map=Un_mapped[i]))
                        break
                    else:
                        idx = (idx + 1) % N
        return In_mapped

    def IV_mapped(self, In, Un, start):
        In, Un = self.sapxep_IV(In = In, Un = Un)
        if self.plot == True:
            plt.plot(Un, In, linestyle='-', color='b', linewidth=3)
            plt.title(f'V-I trajectory tại {start} sau khi sap xep {self.file}')
            plt.ylabel('current (A)')
            plt.xlabel('Voltage (V)')
            plt.show()
        In, Un = self.smooth_vi(Un = Un, In = In)
        In, Un = self.sapxep_I(In = In, Un = Un)
        if self.plot == True:
            plt.plot(Un, In, linestyle='-', color='b', linewidth=3)
            plt.title(f'V-I trajectory tại {start} sau khi lam muot {self.file} là')
            plt.ylabel('current (A)')
            plt.xlabel('Voltage (V)')
            plt.show()
        Un_mapped = self.map_U(Un)
        In_mapped = self.map_I(In = In, Un = Un, Un_mapped = Un_mapped)
        if self.plot == True:
            plt.scatter(Un_mapped, In_mapped, color='r', s = 50)
            plt.title(f'V-I trajectory tại {start} sau khi map {self.file} là')
            plt.ylabel('current (A)')
            plt.xlabel('Voltage (V)')
            plt.show()
        return In_mapped, Un_mapped
    
    def return_IV_mapped(self, start):
        In1 = self.In[start:start+self.point_per_cycle * self.num_cycle]
        Un1 = self.Un[start:start+self.point_per_cycle * self.num_cycle]
        Power_mean = np.mean(np.abs(Un1 * In1))

        In_mapped, Un_mapped = self.IV_mapped(In = In1, Un = Un1, start = start)
        return In_mapped, Un_mapped, Power_mean

    def return_IV_image(self, start, plot = False):
        In_mapped, Un_mapped, Power_mean = self.return_IV_mapped(start = start)
        matrix = self.flip_I_V(In=In_mapped, Un=Un_mapped)
        if plot:
            matrix = 255 - matrix.reshape((2*self.N + 1, 2*self.N + 1))
            matrix = np.array(matrix, dtype=np.uint8)
            plt.imshow(matrix, cmap="gray", origin="lower")
            plt.title(f"{self.file}")
            plt.show()
        return matrix, Power_mean

    def flip_I_V(self, In, Un):
        In = np.array(In)
        Un = np.array(Un)
        min_index = np.argmin(Un)
        if In[min_index] > 0:
            In = -In
        matrix = I_V_image_single_cycle_2(In=In, Un=Un)
        upper_matrix = np.sum(matrix[16:34, :])
        lower_matrix = np.sum(matrix[:16, :])
        if lower_matrix > upper_matrix:
            matrix = np.flip(matrix, axis=(0, 1))
        return matrix.flatten()
    
    def plot_Un(self, start):
        In1 = self.In[start : start + self.point_per_cycle * self.num_cycle]
        Un1 = self.Un[start : start + self.point_per_cycle * self.num_cycle]

        # Hàm IV_mapped trả về 4 giá trị
        In_mapped, Un_mapped, In_or, Un_or= self.IV_mapped(In=In1, Un=Un1, start=start)
        # Scatter original
        plt.scatter(Un_or, In_or, color='red',  label='original')
        # Scatter mapped
        plt.scatter(Un_mapped, In_mapped, color='blue', label='mapped')

        plt.xlabel("Current (In)")
        plt.ylabel("Voltage (Un)")
        plt.title("I-V Original vs Mapped")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
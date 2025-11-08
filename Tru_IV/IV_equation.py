import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from .iv_image_nam_ver import I_V_image_ver_1, I_V_image_ver_3, I_V_image_single_cycle, I_V_image_single_cycle_2
from .Map_IV import MapIV_2
    
class TruIV:
    def __init__(self, file_minuend, file_subtrahend, N = 16, num_map = 200, plot = False, num_cycle = 20):
        self.file_minuend = file_minuend
        self.file_subtrahend = file_subtrahend
        self.N = N
        self.num_map = num_map
        self.mapiv_minuend = MapIV_2(file_minuend, num_map=self.num_map, num_cycle=num_cycle)
        self.mapiv_subtrahend = MapIV_2(file_subtrahend, num_map=self.num_map, num_cycle=num_cycle)
        self.plot = plot

    def IV_minus_plot(self, start_file_minuend, start_file_subtrahend):
        In_mapped_minuend, Un_mapped_minuend, Power_minuend = self.mapiv_minuend.return_IV_mapped(start_file_minuend)
        In_mapped_subtrahend, Un_mapped_subtrahend, Power_subtrahend = self.mapiv_subtrahend.return_IV_mapped(start_file_subtrahend)
        
        In_minus = np.array(In_mapped_minuend) - np.array(In_mapped_subtrahend)
        Un_minus = (np.array(Un_mapped_minuend) + np.array(Un_mapped_subtrahend)) / 2
        delta_Power = np.abs(Power_minuend - Power_subtrahend)
        
        if self.plot:
            plt.scatter(Un_minus, In_minus, linestyle='-', color='g')
            plt.title(f'V-I trajectory (subtraction result)')
            plt.ylabel('current (A)')
            plt.xlabel('Voltage (V)')
            plt.show()
        
        return In_minus, Un_minus, delta_Power

    def IV_minus_image_plot(self, start_file_minuend, start_file_subtrahend):
        In_mapped_minuend, Un_mapped_minuend, _ = self.mapiv_minuend.return_IV_mapped(start_file_minuend)
        In_mapped_subtrahend, Un_mapped_subtrahend, _ = self.mapiv_subtrahend.return_IV_mapped(start_file_subtrahend)
        
        In_minus = np.array(In_mapped_minuend) - np.array(In_mapped_subtrahend)
        Un_minus = - (np.array(Un_mapped_minuend) + np.array(Un_mapped_subtrahend)) / 2
        
        def plot_IV_image(In, Un, title):
            matrix = I_V_image_single_cycle(In=In, Un=Un, N=self.N)
            matrix = 255 - matrix.reshape((2*self.N+1, 2*self.N+1))
            plt.imshow(matrix, cmap="gray", origin="lower")
            plt.title(title)
            plt.show()
        if self.plot:
            plot_IV_image(In_mapped_minuend, Un_mapped_minuend, f'V-I image minuend: {self.file_minuend}')
            plot_IV_image(In_mapped_subtrahend, Un_mapped_subtrahend, f'V-I image subtrahend: {self.file_subtrahend}')
            plot_IV_image(In_minus, Un_minus, 'V-I image: subtraction result')

    def IV_minus_img(self, start_file_minuend, start_file_subtrahend):
        In_mapped_minuend, Un_mapped_minuend, Power_minuend = self.mapiv_minuend.return_IV_mapped(start_file_minuend)
        In_mapped_subtrahend, Un_mapped_subtrahend, Power_subtrahend = self.mapiv_subtrahend.return_IV_mapped(start_file_subtrahend)

        In_minus = np.array(In_mapped_minuend) - np.array(In_mapped_subtrahend)
        Un_minus = (np.array(Un_mapped_minuend) + np.array(Un_mapped_subtrahend)) / 2
        matrix_minus = self.flip_I_V(In = In_minus, Un = Un_minus)
        
        delta_Power = np.abs(Power_minuend - Power_subtrahend)
        
        if self.plot:
            matrix_minus = 255 - matrix_minus.reshape((2*self.N + 1, 2*self.N + 1))
            plt.imshow(matrix_minus, cmap="gray", origin="lower")
            plt.title(f'minus {self.file_minuend} - {self.file_subtrahend}')
            plt.show()

            print(f"Delta Power của 2 đoạn là {delta_Power}")

        return matrix_minus.flatten(), delta_Power
    
    def flip_I_V(self, In, Un):
        In = np.array(In)
        Un = np.array(Un)
        min_index = np.argmin(Un)
        if In[min_index] > 0:
            In = -In
        matrix = I_V_image_single_cycle_2(In=In, Un=Un)
        upper_matrix = np.sum(matrix[16:33, :])   # lấy các hàng chỉ số 16 → 32
        lower_matrix = np.sum(matrix[:16, :])
        if lower_matrix > upper_matrix:
            matrix = np.flip(matrix, axis=(0, 1))
        return matrix

class delta_IV_equation:
    def __init__(self, plot = False):
        self.plot = plot
        self.N = 16
        self.train_df = pd.read_csv("/home/mylab-nilm/Code/hainam/I_V_combining/data/data_train_test/train_data/train_data.csv")
        self.test_df = pd.read_csv("/home/mylab-nilm/Code/hainam/I_V_combining/data/data_train_test/test_data/test_event_only.csv")
        self.matrix_dict = self.create_matrix_dict()

    def create_matrix_dict(self):
        matrix_dict = {}
        for label in range(7):  # các nhãn 0 → 6
            filtered_df = self.train_df[self.train_df["Label"] == label]

            row = filtered_df.iloc[1]
            matrix = 255 - row.iloc[4:].values.reshape((33, 33))
            power = row.iloc[3]

            matrix_dict[label] = {
                "matrix": matrix,
                "power": power
            }

        return matrix_dict

    def delta_IV(self, matrix1, matrix2, arg1 = 1, arg2 = 0.05, arg3 = 0.1):
        difference = 0
        for i in range(matrix1.shape[0]):
            for j in range(matrix1.shape[1]):
                if (matrix1[i][j] != 0 and matrix2[i][j] != 0):
                    difference += arg2 * abs(matrix1[i][j] - matrix2[i][j])
                elif matrix1[i][j] != 0 and matrix2[i][j] == 0:
                    neighbors = []
                    if i > 0:
                        neighbors.append(matrix2[i-1][j])
                    if i < matrix1.shape[0] - 1:
                        neighbors.append(matrix2[i+1][j])
                    if j > 0:
                        neighbors.append(matrix2[i][j-1])
                    if j < matrix1.shape[0] - 1:
                        neighbors.append(matrix2[i][j+1])
                    v1 = matrix1[i][j]
                    v2 = matrix2[i][j]
                    if any(n == 0 for n in neighbors):
                        difference += arg3 * abs(v1 - v2)
                    else:
                        difference += arg1 * abs(v1 - v2)
                elif matrix1[i][j] == 0 and matrix2[i][j] != 0:
                    neighbors = []
                    if i > 0:
                        neighbors.append(matrix1[i-1][j])
                    if i < matrix1.shape[0] - 1:
                        neighbors.append(matrix1[i+1][j])
                    if j > 0:
                        neighbors.append(matrix1[i][j-1])
                    if j < matrix1.shape[0] - 1:
                        neighbors.append(matrix1[i][j+1])
                    v1 = matrix1[i][j]
                    v2 = matrix2[i][j]
                    if any(n == 0 for n in neighbors):
                        difference += arg3 * abs(v1 - v2)
                    else:
                        difference += arg1 * abs(v1 - v2)
        return difference/255
    
    def test_result(self):
        y_true = self.test_df["Label"].values
        y_pred = []
        for index, row in self.test_df.iterrows():
            matrix_minus = row.iloc[9:].values
            power = row.iloc[8]
            label = self.return_label(power, matrix_minus)
            y_pred.append(label)

        return y_true, y_pred

    def return_label(self, delta_Power, matrix_minus):
        matrix_minus = 255 - matrix_minus.reshape((2*self.N + 1, 2*self.N + 1))
        score_arr = []
        for label in range(0, 7):
            stored_matrix = self.matrix_dict[label]["matrix"]
            stored_power = self.matrix_dict[label]["power"]
            delta_iv = self.delta_IV(matrix_minus, stored_matrix)
            delta_power = abs(delta_Power - stored_power) / 100
            score = delta_iv + delta_power
            score_arr.append((label, score))

        best_label, _ = min(score_arr, key=lambda x: x[1])
        return best_label
        

                

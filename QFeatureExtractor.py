from Utilis.NILM_Utilis import (CycleInterpolator, align_phase, calc_prms, 
                                smooth_savgol, flip_ui_image, plot_to_bw_image_with_gaussian_dots)
import numpy as np
import config
import pandas as pd
import matplotlib.pyplot as plt

class QFeatureExtracter:
    def __init__(self,I_raw, U_raw,SAMPLING_RATE = 1000,FREQUENCY = 50,IMAGE_CYCLE_DURATION = 2, INTERP_FACTOR = 10):
        self.I_raw = I_raw
        self.U_raw = U_raw
        self.sample_per_circle = int(SAMPLING_RATE / FREQUENCY)
        self.circle_per_image = int(IMAGE_CYCLE_DURATION * FREQUENCY)
        self.sample_per_image = int(self.sample_per_circle * self.circle_per_image)
        self.interp_factor = INTERP_FACTOR
        
    def getFeature(self,start1,start2):
        i1 = self.I_raw[start1 : start1 + self.sample_per_image]
        u1 = self.U_raw[start1 : start1 + self.sample_per_image]
        i2 = self.I_raw[start2 : start2 + self.sample_per_image]
        u2 = self.U_raw[start2 : start2 + self.sample_per_image]

        if len(i1) < self.sample_per_circle or len(i2) < self.sample_per_circle:
            return [],0

        last_circle = CycleInterpolator(self.sample_per_circle, self.interp_factor)
        last_circle.update_batch(i1, u1)
        current_circle = CycleInterpolator(self.sample_per_circle, self.interp_factor)
        current_circle.update_batch(i2, u2)

        u_last, i_last = last_circle.get_average()
        u_cur, i_cur = current_circle.get_average()

        _, best_shift = align_phase(u_cur, u_last)
        i_last_aligned = np.roll(i_last, -int(best_shift))
        i_res = (i_cur - i_last_aligned)
        u_res = u_cur
        u_res = smooth_savgol(u_res)
        i_res = smooth_savgol(i_res)
        
        delta_p_mean = abs(calc_prms(i2,u2) - calc_prms(i1,u1))
        
        img_np = plot_to_bw_image_with_gaussian_dots(u_res, i_res, config.IMAGE_SIZE, config.IMAGE_SIZE,config.IMG_DOT_RADIUS,config.IMG_DOT_ALPHA)
        img_np = flip_ui_image(img_np)
        
        return img_np,delta_p_mean
    
# Mẫu dùng thử
# csv_path = r"ElectricDatas\MyNewData\data_12_mayep_tulanh_event_on_quat.csv"
# df = pd.read_csv(csv_path)
# Power = df["Power"].values
# I_raw = df["In"].values
# U_raw = df["Un"].values
# fe = QFeatureExtracter(I_raw,U_raw)
# img,p =fe.getFeature(10000,50000)
# plt.title("P = {p}")
# plt.imshow(img, cmap='gray')
# plt.show()
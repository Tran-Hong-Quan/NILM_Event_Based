import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from .IV_equation import TruIV
import torch
from .CNN_model import CNN_TwoBranch_1
current_dir = os.path.dirname(os.path.abspath(__file__))

class FullSystem:
    def __init__(self):
        pass

    def minus_and_predict(self, file, start_1, start_2, plot = False):
        tru_iv = TruIV(file_minuend = file, file_subtrahend = file, plot = plot)
        matrix_minus, Power_mean = tru_iv.IV_minus_img(start_file_minuend = start_1, start_file_subtrahend = start_2)
        Power_mean = np.array(Power_mean).reshape(-1, 1)
        matrix_minus = np.array(matrix_minus).reshape(-1, 1, 33, 33)
        X1_tensor = torch.tensor(Power_mean, dtype=torch.float32)
        X2_tensor = torch.tensor(matrix_minus, dtype=torch.float32)
        model = CNN_TwoBranch_1(num_classes=7)
        checkpoint_path = os.path.join(current_dir, "cnn_model_1.pth")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            outputs = model(X1_tensor, X2_tensor)
            preds = torch.argmax(outputs, dim=1)
            return preds
    
    def predict(self, file, start_1, start_2, plot = False):
            y_pred = self.minus_and_predict(file, start_1, start_2, plot)
            device_dict = {
                7: "null",
                0: "quat",
                1: "tulanh",
                2: "mayep",
                3: "maysay",
                4: "sacmt",
                5: "mayhutbui",
                6: "manhinh"
            }
            pred_label = int(y_pred[0])   # lấy số dự đoán
            return device_dict.get(pred_label, "null")
import numpy as np
class KalmanFilter:
    def __init__(self, Q=1e-3, R=100, initial_estimate=0, initial_covariance=1):
        self.Q = Q  # Nhiễu quá trình
        self.R = R  # Nhiễu đo lường
        self.x_est = initial_estimate  # Trạng thái ước lượng ban đầu
        self.P_est = initial_covariance  # Hiệp phương sai ban đầu
    
    def update(self, measurement):
        # Dự đoán (Prediction Step)
        x_pred = self.x_est
        P_pred = self.P_est + self.Q
        
        # Cập nhật (Update Step)
        K = P_pred / (P_pred + self.R)  # Hệ số Kalman
        self.x_est = x_pred + K * (measurement - x_pred)
        self.P_est = (1 - K) * P_pred
        
        return self.x_est
    
    def filter(self, data):
        filtered_data = np.zeros(len(data))
        for i, measurement in enumerate(data):
            filtered_data[i] = self.update(measurement)
        return filtered_data
    
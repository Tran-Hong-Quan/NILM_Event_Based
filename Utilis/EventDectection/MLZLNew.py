import numpy as np

class MLZLNew:
    def __init__(self, window_size=6, threshold=30):
        self.window_size = window_size 
        self.threshold = threshold  
        self.last_winSize = len(self.buffer)
        self.buffer = []
        self.meanLast = 0

    def update(self, power_value):
        self.buffer.append(power_value)

        if len(self.buffer) < 2 * self.window_size:
            return 0
        before_window = self.buffer[:self.window_size]
        after_window = self.buffer[-self.window_size:]
        mean_before = np.mean(before_window)
        mean_after = np.mean(after_window)
        delta = mean_after - mean_before
        delta2 = mean_after - self.meanLast
        event = 0
        if abs(delta) < abs(delta2):
            delta = delta2
        if abs(delta) >= self.threshold:
            if(delta > 0):
                event = 1
            else:
                event = -1
        self.buffer = []
        self.meanLast = mean_before
        
        return event

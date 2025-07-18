import numpy
from KalmanFilter import KalmanFilter
from MLZLNew import MLZLNew
from WAMMA import WAMMA

def mean(arr):
    return numpy.mean(arr)

class QuanDetector:
    def __init__(self, event_sampling_rate=100,
                 wamma_window_sec=10, wamma_edge_sec=3,
                 low_dec_sampling_rate=1, low_dec_window_sec=30,
                 wamma_p_threshold=30, wamma_r_threshold=1, low_dec_threshold=10,
                 event_time_limit_dif=30, event_time_limit_sam=1, init_power=0,
                 kalman_Q = 0.01, kalman_R = 100):
        
        # Sampling and window setup
        self.EVENT_SAMPLING_RATE = event_sampling_rate
        self.LOW_DEC_SAMPLING_RATE = low_dec_sampling_rate
        self.low_dec_buffer_size = int(self.EVENT_SAMPLING_RATE / low_dec_sampling_rate)
        self.low_dec_window_size = int(low_dec_window_sec // low_dec_sampling_rate // 2)
        
        # Thresholds and timing
        self.wamma_p_threshold = wamma_p_threshold
        self.wamma_r_threshold = wamma_r_threshold
        self.low_dec_threshold = low_dec_threshold
        self.EVENT_TIME_LIMIT_DIF = event_time_limit_dif
        self.EVENT_TIME_LIMIT_SAM = event_time_limit_sam

        # WAMMA parameters
        self.wamma = WAMMA(int(wamma_window_sec * self.EVENT_SAMPLING_RATE),
                           int(wamma_edge_sec * self.EVENT_SAMPLING_RATE),
                           wamma_p_threshold, wamma_r_threshold)
        
        # Kalman filter
        self.kalman = KalmanFilter(kalman_Q, kalman_R, init_power, 1)

        # Low decision detector
        self.low_dec = MLZLNew(self.low_dec_window_size, self.low_dec_threshold)

        # Internal states
        self.avg_win = []
        self.indexFromLastEvent = 0
        self.lastEventType = -1

    def update(self, value):
        event_output = 0  # No event by default
        f = self.kalman.update(value)
        self.filtedSig = f
        ew = self.wamma.update(value)
        e_lo = 0
        self.avg_win.append(f)

        if len(self.avg_win) == self.low_dec_buffer_size:
            avg_val = mean(self.avg_win)
            e_lo = self.low_dec.update(avg_val)
            self.avg_win = []

        self.indexFromLastEvent += 1
        currentEventType = -1
        windowDuration = 0

        if ew != 0:
            event_output = ew
            currentEventType = 0
            windowDuration = 1 / self.EVENT_SAMPLING_RATE * self.wamma.last_winSize
        elif e_lo != 0:
            event_output = e_lo
            currentEventType = 1
            windowDuration = 1 / self.LOW_DEC_SAMPLING_RATE * self.low_dec.last_winSize

        should_emit = (
            (event_output != 0 and 
            ((self.indexFromLastEvent / self.EVENT_SAMPLING_RATE > self.EVENT_TIME_LIMIT_DIF and currentEventType != self.lastEventType) or
             (self.indexFromLastEvent / self.EVENT_SAMPLING_RATE > self.EVENT_TIME_LIMIT_SAM and currentEventType == self.lastEventType)))
            or self.lastEventType == -1)
            
        if should_emit:
            self.indexFromLastEvent = 0
            self.lastEventType = currentEventType
            self.buffer = []
        else:
            event_output = 0

        return event_output, windowDuration
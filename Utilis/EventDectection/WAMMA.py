class WAMMA:
    def __init__(self, Nw, Nm,P_thr,r_thr):
        self.Nw = Nw
        self.Nm = Nm
        self.P_thr = P_thr
        self.r_thr = r_thr
        self.window = []
        self.createWindow()

    def createWindow(self):
        self.Nw_w = self.Nw
        self.N_Lm = self.Nm
        self.N_Rm = self.Nm
        self.delta_P = 0
        self.last_winSize = len(self.window)
        self.window.clear()
        self.P_thre_w = self.P_thr

    def adjustWindows(self):
        self.Nw_w += 1
        self.N_Lm -= 1

    def update(self, value):
        self.window.append(value)

        if len(self.window) < self.Nw_w:
            return 0
        
        # Adaptive threshold updating
        di = 0
        u = sum(self.window) / self.Nw_w
        for o in self.window:
            di += (o - u)**2
        di /= self.Nw_w
        di = di**.5
        self.P_thre_w = max(self.P_thr,self.r_thr * di)
        
        # Margins Adjustment
        t = self.P_thre_w * .1

        delta_Pl = abs(self.window[self.N_Lm - 1] - self.window[0])
        if(self.N_Lm <= 0):
            delta_Pl = 0
        if delta_Pl > t and self.N_Lm > 2:
            #print("Left Adj 1")
            self.adjustWindows()
            return 0
        
        delta_Pr = abs(self.window[self.Nw_w - self.N_Rm] - self.window[self.Nw_w - 1])
        if delta_Pr > t:
            #print("Right Adj 1")
            self.adjustWindows()
            return 0
        
        Si = 0
        for i in range(1,int(self.N_Lm / 2)):
            di =  WAMMA.sgn(self.window[i] - self.window[i-1])
            Si += di
            PrSi = abs(Si) / i
            if PrSi > 0.6 and self.N_Lm > 2:
                #print("Left Adj 2")
                self.adjustWindows()
                return 0
                
        Si = 0
        for i in range(self.Nw_w - self.N_Rm,int(self.Nw_w - self.N_Rm / 2)):
            di = WAMMA.sgn(self.window[i] - self.window[i-1])
            Si += di
            PrSi = abs(Si) / i
            if PrSi > 0.6:
                #print("Right Adj 2")
                self.adjustWindows()
                return 0
            
        # Caculate event
        if self.N_Lm <= 0:
            ul = self.window[0]
        else :
            ul = sum(self.window[:self.N_Lm]) / self.N_Lm
        ur = sum(self.window[-self.N_Rm:]) / self.N_Rm
        delta_P = ur - ul
        
        sgt = 0
        if delta_P > self.P_thre_w:
            sgt = 1
        elif delta_P < -self.P_thre_w:
            sgt = -1
    
        self.createWindow()

        return sgt
    
    def sgn(value):
        if(value > 0.001):
            return 1
        if(value < -0.001):
            return -1
        return 0

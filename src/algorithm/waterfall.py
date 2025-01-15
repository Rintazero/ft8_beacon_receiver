import numpy as np
import scipy

class StftCore:
    def __init__(self, ):
        pass

    def run(self):
        pass

class Waterfall:
    def __init__(self, 
                 sampleRate: np.uint16,                     # 采样率
                 windowSize: np.uint16,                     # 窗口大小
                 overlap: np.uint16,                        # 重叠
                 windowType: str,                           # 窗口类型
                 nfft: np.uint16,                           # fft点数
                 freqRange: tuple[np.uint16, np.uint16],    # 频率范围
                 dbRange: tuple[np.uint16, np.uint16],):    # 分贝范围
        
        self.sampleRate = sampleRate
        self.windowSize = windowSize
        self.overlap = overlap
        self.windowType = windowType
        self.nfft = nfft
        self.freqRange = freqRange
        self.dbRange = dbRange

        self.tframe = np.zeros(windowSize)
        self.fframe = np.zeros(nfft)
        self.data = np.zeros((self.nfft, 1))

    def run(self):
        pass


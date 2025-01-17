import numpy as np
import scipy
import matplotlib.pyplot as plt

# basic variables
time_osr = int(2)
freq_osr = int(2)
NSyncSym = 7
NDataSym = 58
waterfall_freq_range = (500, 3050)

# load raw data
rawSignalInfo = scipy.io.loadmat("./data/raw/waveInfo.mat")
fs = rawSignalInfo.get("Fs")[0][0]
f0 = rawSignalInfo.get("F0")[0][0]
waveRe = rawSignalInfo.get("waveRe")[0]
waveIm = rawSignalInfo.get("waveIm")[0]
nsamples = len(waveRe)
wave = np.complex64(waveRe + waveIm)
SymBin = rawSignalInfo.get("SymBin")[0][0]
SymT = rawSignalInfo.get("SymT")[0][0]

# 构造 (线性)频移 载波

fShift_t0_Hz = np.float64(0) # 初始频偏( \delta f(t=0) )
fShift_k_Hzpsample = np.float64(2000/nsamples) # 频偏变化率
print("fShift_t0_Hz:",fShift_t0_Hz,"fShift_k_Hzpsample:",fShift_k_Hzpsample)

shiftCarrier = np.exp(2j*np.pi*fShift_t0_Hz*np.arange(nsamples)/fs + 2j*np.pi*fShift_k_Hzpsample*np.arange(nsamples)**2/(2*fs))

# 频移载波与原始信号相乘
# wave_shift = np.complex64( -1 * waveRe * np.real(shiftCarrier) + waveIm * np.imag(shiftCarrier))
wave_shift = (waveRe + (1j) * waveIm) * shiftCarrier

# 高斯噪声
SNR = -10
wave_power = np.mean(np.abs(wave)**2)
noise_power = wave_power / (10**(SNR/10))

print("wave_power:",wave_power,"noise_power:",noise_power)

noise = np.random.normal(0, np.sqrt(noise_power), nsamples)
wave_noise = np.complex64(wave_shift + noise)

# stft
nsps = int(SymT*fs)
nfft = int(fs*SymT*freq_osr)
tlength = len(wave_noise) / fs
hann_window = scipy.signal.windows.hann(M=nfft, sym=False)
SFT = scipy.signal.ShortTimeFFT(win = hann_window, hop = nsps//time_osr, fs = fs, fft_mode = "onesided")
sx = SFT.stft(np.real(wave_noise))
freq_range = waterfall_freq_range
freqs = np.linspace(0, fs/2, sx.shape[0])
freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
sx_filtered = sx[freq_mask, :]
sx_filtered_real = np.real(sx_filtered)
sx_filtered_imag = np.imag(sx_filtered)
sx_filtered_power = sx_filtered_real**2 + sx_filtered_imag**2
sx_filtered_db = 10 * np.log10(sx_filtered_power)
max_freq_indices =  np.argmax(np.abs(sx_filtered), axis=0)
max_freqs = freq_range[0] + freqs[max_freq_indices]


# 构造 同步序列
SyncSeq = (np.array([3,1,4,0,6,5,2]) + 1) * SymBin
syncCorrelationSeq = np.zeros( (3*NSyncSym + NDataSym) * time_osr )
for i in range(len(syncCorrelationSeq)):
    if i < NSyncSym * time_osr:
        syncCorrelationSeq[i] = SyncSeq[i//time_osr]
    elif i >= (NSyncSym + NDataSym//2) * time_osr and i < (2*NSyncSym + NDataSym//2) * time_osr:
        syncCorrelationSeq[i] = SyncSeq[(i-(NSyncSym + NDataSym//2) * time_osr)//time_osr]
    elif i >= (2*NSyncSym + NDataSym) * time_osr:
        syncCorrelationSeq[i] = SyncSeq[(i-(2*NSyncSym + NDataSym) * time_osr)//time_osr]
    else:
        syncCorrelationSeq[i] = 0


# 相关
syncCorrelation = np.correlate(max_freqs - np.min(max_freqs), syncCorrelationSeq, mode='full')

correlationPeakIndex = np.argmax(syncCorrelation)
correlationPeakTimeBlockIndex = correlationPeakIndex - len(syncCorrelationSeq)
print("correlationPeakTimeBlockIndex:",correlationPeakTimeBlockIndex)

# 频偏估计
fSyncSegment = np.zeros([3,NSyncSym*time_osr])

fSyncSegment[0] = max_freqs[correlationPeakTimeBlockIndex + range(NSyncSym*time_osr)]
fSyncSegment[1] = max_freqs[correlationPeakTimeBlockIndex + (NSyncSym + NDataSym//2)*time_osr + range(NSyncSym*time_osr)]
fSyncSegment[2] = max_freqs[correlationPeakTimeBlockIndex + (2*NSyncSym + NDataSym)*time_osr + range(NSyncSym*time_osr)]

fShift_est_k_Hzpsample = np.mean(np.mean(fSyncSegment[1]-fSyncSegment[0]) + np.mean(fSyncSegment[2]-fSyncSegment[1]))/((NSyncSym+NDataSym//2)*time_osr)/nsps
print("fShift_est_k_Hzpsample:",fShift_est_k_Hzpsample)
print("Linear Frequency Deviation Rate Error:{}Hz".format((fShift_k_Hzpsample - fShift_est_k_Hzpsample)*nsamples))

# 频偏补偿
CompensationCarrier = np.exp(-2j*np.pi*fShift_est_k_Hzpsample*np.arange(nsamples)**2/(2*fs))
wave_compensated = wave_noise * CompensationCarrier
wave_compensated_sx = SFT.stft(np.real(wave_compensated))
compensated_freq_range = (waterfall_freq_range[0], waterfall_freq_range[0]+200)
compensated_freq_mask = (freqs >= compensated_freq_range[0]) & (freqs <= compensated_freq_range[1])
wave_compensated_sx_filtered = wave_compensated_sx[compensated_freq_mask, :]
wave_compensated_sx_filtered_real = np.real(wave_compensated_sx_filtered)
wave_compensated_sx_filtered_imag = np.imag(wave_compensated_sx_filtered)
wave_compensated_sx_filtered_power = wave_compensated_sx_filtered_real**2 + wave_compensated_sx_filtered_imag**2
wave_compensated_sx_filtered_db = 10 * np.log10(wave_compensated_sx_filtered_power)






# 可视化

plt.figure(num=1, figsize=(10, 6))
plt.imshow(sx_filtered_db, aspect='auto', origin='lower', 
           extent=[0, tlength, freq_range[0], freq_range[1]])
plt.colorbar(label='Magnitude(dB)')
plt.title('Short-Time Fourier Transform (STFT) Magnitude(dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.grid()
plt.show()

plt.figure(num=2, figsize=(10, 6))
plt.plot(np.linspace(0, tlength, len(max_freqs)), max_freqs, marker='o', linestyle='-',zorder=1)
plt.title('Maximum Frequencies Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid()
plt.xlim(0, tlength)
plt.ylim(freq_range[0], freq_range[1])
highlight_index = correlationPeakTimeBlockIndex + list(range(NSyncSym*time_osr))
plt.scatter(highlight_index * SymT/time_osr, max_freqs[highlight_index], color='red', marker='o',zorder=2)
plt.grid()
plt.show()

plt.figure(num=3, figsize=(10, 6))
plt.plot(syncCorrelation, marker='o', linestyle='-', color='b')
plt.title('Synchronization Correlation')
plt.xlabel('Sample Index')
plt.ylabel('Correlation Value')
plt.grid()
plt.xlim(0, len(syncCorrelation))
plt.ylim(np.min(syncCorrelation), np.max(syncCorrelation))
plt.show()

plt.figure(num=4, figsize=(10, 6))
plt.imshow(wave_compensated_sx_filtered_db, aspect='auto', origin='lower', 
           extent=[0, tlength, compensated_freq_range[0], compensated_freq_range[1]])
plt.colorbar(label='Magnitude (dB)')
plt.title('Wave Compensated STFT Magnitude (Filtered)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.grid()
plt.show()

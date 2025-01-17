import numpy as np
import scipy
import matplotlib.pyplot as plt

# basic variables
time_osr = int(2)
freq_osr = int(2)

# load raw data
rawSignalInfo = scipy.io.loadmat("./data/raw/waveInfo.mat")
fs = rawSignalInfo.get("Fs")[0][0]
f0 = rawSignalInfo.get("F0")[0][0]
waveRe = rawSignalInfo.get("waveRe")[0]
waveIm = rawSignalInfo.get("waveIm")[0]
nsamples = len(waveRe)
wave = np.complex64(waveRe + waveIm)

# check rawSignalInfo
print(waveRe,type(waveRe),np.shape(waveRe))

# 构造 (线性)频移 载波

fShift_t0_Hz = np.float64(50) # 初始频偏( \delta f(t=0) )
fShift_k_Hzpsample = np.float64(100/nsamples) # 频偏变化率

shiftCarrier = np.exp(2j*np.pi*fShift_t0_Hz*np.arange(nsamples)/fs + 2j*np.pi*fShift_k_Hzpsample*np.arange(nsamples)**2/(2*fs))

# 频移载波与原始信号相乘
wave_shift = np.complex64( -1 * waveRe * np.real(shiftCarrier) + waveIm * np.imag(shiftCarrier))

# 高斯噪声
SNR = -15
wave_power = np.mean(wave**2)
noise_power = wave_power / (10**(SNR/10))
noise = np.random.normal(0, np.sqrt(noise_power), nsamples)
wave_noise = np.complex64(wave_shift + noise)

# 
nsps = int(0.16*fs)
nfft = int(fs*0.16*freq_osr)

tlength = len(waveRe) / fs

hann_window = scipy.signal.windows.hann(M=nfft, sym=False)
SFT = scipy.signal.ShortTimeFFT(win = hann_window, hop = nsps//time_osr, fs = fs, fft_mode = "onesided")
sx = SFT.stft(np.real(wave_noise))

freq_range = (500, 850)
freqs = np.linspace(0, fs/2, sx.shape[0])
freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
sx_filtered = sx[freq_mask, :]

sx_filtered_real = np.real(sx_filtered)
sx_filtered_imag = np.imag(sx_filtered)
sx_filtered_power = sx_filtered_real**2 + sx_filtered_imag**2

sx_filtered_db = 10 * np.log10(sx_filtered_power)
plt.figure(figsize=(10, 6))
plt.imshow(sx_filtered_db, aspect='auto', origin='lower', 
           extent=[0, tlength, freq_range[0], freq_range[1]])
plt.colorbar(label='Magnitude')
plt.title('Short-Time Fourier Transform (STFT) Magnitude (450Hz to 650Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.grid()
plt.show()


max_freq_indices =  np.argmax(np.abs(sx_filtered), axis=0)
max_freqs = freq_range[0] + freqs[max_freq_indices]


plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, tlength, len(max_freqs)), max_freqs, marker='o', linestyle='-')
plt.title('Maximum Frequencies Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid()
plt.xlim(0, tlength)
plt.ylim(freq_range[0], freq_range[1])

plt.show()
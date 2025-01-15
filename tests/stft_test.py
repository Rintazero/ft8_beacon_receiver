import numpy as np
import scipy
import matplotlib.pyplot as plt

# basic variables
time_osr = int(4)
freq_osr = int(4)

fs, signal = scipy.io.wavfile.read('./data/raw/ft8_fs20k_f0_550_id_1.wav')
# signal = signal.astype(np.double)

nsps = int(0.16*fs)
nfft = int(fs*0.16*freq_osr)

tlength = len(signal) / fs

print("signal_info:fs = {}Hz, signal_length = {}samples, tlength = {}s, dtype = {}".format(fs, len(signal), tlength, signal.dtype))
print("nsps = {}, dtype = {}".format(nsps, type(nsps)))

max_signal_value = np.max(signal)
print("Maximum value of the signal: {}".format(max_signal_value))

hann_window = scipy.signal.windows.hann(M=nfft, sym=False)
print("hann_window_info: dtype = {}, shape = {}".format(hann_window.dtype, hann_window.shape))

SFT = scipy.signal.ShortTimeFFT(win = hann_window, hop = nsps//time_osr, fs = fs, fft_mode = "onesided")

sx = SFT.stft(signal)
print("dtype = {}, sx_shape = {}".format(sx.dtype, sx.shape))

freq_range = (450, 650)
freqs = np.linspace(0, fs/2, sx.shape[0])
freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
sx_filtered = sx[freq_mask, :]

sx_filtered_db = 20 * np.log10(np.abs(sx_filtered) + 1e-10)  # Adding a small value to avoid log(0)
# print("Filtered STFT in dB: dtype = {}, shape = {}".format(sx_filtered_db.dtype, sx_filtered_db.shape))


plt.figure(figsize=(10, 6))
plt.imshow(np.abs(sx_filtered_db), aspect='auto', origin='lower', 
           extent=[0, tlength, freq_range[0], freq_range[1]])
plt.colorbar(label='Magnitude')
plt.title('Short-Time Fourier Transform (STFT) Magnitude (450Hz to 650Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.grid()
plt.show()

# Find the frequency corresponding to the maximum value at each time slice
max_freq_indices =  np.argmax(np.abs(sx_filtered), axis=0)
max_freqs = freq_range[0] + freqs[max_freq_indices]

# Visualize the max_freqs
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, tlength, len(max_freqs)), max_freqs, marker='o', linestyle='-')
plt.title('Maximum Frequencies Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid()
plt.xlim(0, tlength)
plt.ylim(freq_range[0], freq_range[1])
#plt.ylim(0, fs/2)
plt.show()





# Visualize the STFT result
# plt.figure(figsize=(10, 6))
# plt.imshow(np.abs(sx), aspect='auto', origin='lower', 
#            extent=[0, tlength, 0, fs/2])
# plt.colorbar(label='Magnitude')
# plt.title('Short-Time Fourier Transform (STFT) Magnitude')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (s)')
# plt.grid()
# plt.show()


# plt.plot(hann_window)
# plt.show()





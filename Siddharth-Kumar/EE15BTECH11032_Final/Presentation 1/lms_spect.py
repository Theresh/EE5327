import scipy
import matplotlib.pyplot as plt
import scipy.io.wavfile
sample_rate, X = scipy.io.wavfile.read('signal_noise.wav')
print (sample_rate, X.shape )
plt.specgram(X, Fs=sample_rate)
plt.show()
sample_rate, X = scipy.io.wavfile.read('noise.wav')
print (sample_rate, X.shape )
plt.specgram(X, Fs=sample_rate)
plt.show()
sample_rate, X = scipy.io.wavfile.read('output_signal_lms.wav')
print (sample_rate, X.shape )
plt.specgram(X, Fs=sample_rate)
plt.show()
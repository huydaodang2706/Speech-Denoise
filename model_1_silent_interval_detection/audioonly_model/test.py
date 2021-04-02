import librosa
import librosa.display
import numpy as np
import transform
# filename = librosa.ex('trumpet')
def real_imag_expand(c_data,dim='new'):
    # dim = 'new' or 'same'
    # expand the complex data to 2X data with true real and image number
    if dim == 'new':
        D = np.zeros((c_data.shape[0],c_data.shape[1],2))
        D[:,:,0] = np.real(c_data)
        D[:,:,1] = np.imag(c_data)
        return D
    if dim =='same':
        D = np.zeros((c_data.shape[0],c_data.shape[1]*2))
        D[:,::2] = np.real(c_data)
        D[:,1::2] = np.imag(c_data)
        return D

y, sr = librosa.load('/home/huydd/NLP/ASR/SentenceSplit/Speech-Denoise/dataset/sos_1/sos_1_0000001.wav',sr=14000)
print(y.shape)
y_stft = transform.fast_stft(y,n_fft=510, hop_length=158, win_length=400)
print(y_stft.shape)

# audio = librosa.stft(y)
# print(audio)
# audio_new = real_imag_expand(audio)
# print(np.count_nonzero(y))
# print(y.shape)
# print(y[10000:10200])
# print(audio.shape)
# print(len(y))
# print(sr)
# 
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# img = librosa.display.specshow(librosa.amplitude_to_db(audio,
#                                                        ref=np.max),
#                                y_axis='log', x_axis='time', ax=ax)
# ax.set_title('Power spectrogram')
# fig.colorbar(img, ax=ax, format="%+2.0f dB")
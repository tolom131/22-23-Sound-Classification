import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import torch

def show_spectrum(amplitude, sampling_rate=44100, hop_length=882):
    fig = plt.figure(figsize = (14,5))
    librosa.display.specshow(amplitude, 
                            sr=sampling_rate, 
                            hop_length=hop_length,
                            x_axis='time',
                            y_axis='log')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')
    plt.show()
    
def play_sound(audio, sr):
    ipd.Audio(audio, rate=sr)     
    
def get_mmm(data):
    print("np.min : %f, np.mean : %f, np.max : %f " %(np.min(data), np.mean(data), np.max(data)))
    

def segment_sound(sound, label, sampling_rate, hop_length, n_mels=64):
    hop_length = int(sampling_rate / 1000 * hop_length)   
    window_size = int(hop_length * 2)
    result_data = np.zeros((60000, window_size))
    result_label = []
    total_count = 0
    less_length = 0
    
    for i, y in enumerate(sound):
        # y = center_crop(y, label[i], sampling_rate*20, len(y)//2)
        if label[i] == 1:
            y = center_crop(y, len(y)//4)
        elif label[i] == 2:
            y = center_crop(y, len(y)//4)

        if len(y) < window_size:
            less_length += 1
            print("sound length is less than 40 ms", less_length)
            continue
        
        D = np.abs(librosa.stft(y, n_fft=window_size, win_length = window_size, hop_length=hop_length, center=False))
        mel_spec = librosa.feature.melspectrogram(S=D, sr=sampling_rate, n_mels=n_mels, hop_length=hop_length, win_length=window_size)
        amplitude = librosa.amplitude_to_db(mel_spec, ref=1.0)
        o_env = librosa.onset.onset_strength(y=y, sr=sampling_rate, S=amplitude)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sampling_rate, hop_length=hop_length, units='samples')
        
        for sample in onset_frames:
            if sample >= len(y) - window_size:
                continue               
            result_data[total_count] = y[sample:sample+window_size]
            total_count += 1
            result_label += [label[i]]
    
    print("the number of total sound segment: %d"%(total_count))
    print("the number of sounds of which length is less than 40ms: %d"%(less_length))
    return result_data[:total_count], np.array(result_label)

def center_crop(audio_data, clip_length):
    # audio_length = len(audio_data)
    # if audio_length > theshold:
    #     print("center cropping", label)
    audio_length = len(audio_data)
    offset = audio_length //2 - clip_length//2
    return audio_data[offset:(offset+clip_length)]

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

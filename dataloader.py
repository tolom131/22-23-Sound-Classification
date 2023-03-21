
import numpy as np
import librosa, librosa.display
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import os

def BGG_dataloader(filepath=None, input_sec=3):
        
    if filepath == None:
        filepath = './dataset/BGGunSound/'
    
    train_path = filepath + 'v3_exp3_train.csv'
    test_path = filepath + 'v3_exp3_test.csv'
    sound_path = filepath + 'gun_sound_wav/'
    name2type_path = filepath + 'name2type.csv'
    
    name2type_df = pd.read_csv(name2type_path)
    name2type_dict = name2type_df.set_index('name').T.to_dict('list')
    name2type = {k : v[0] for (k, v) in name2type_dict.items()} 
  
    train_df = pd.read_csv(train_path)
    assert sorted(list(name2type.keys())) == sorted(list(set(train_df["cate"])))
    test_df = pd.read_csv(test_path)
    assert sorted(list(name2type.keys())) == sorted(list(set(test_df["cate"])))
    
    le = LabelEncoder()
    le.fit(list(name2type.values()))
   
    sampling_rate = 44100
    rclip = RandomClip(sampling_rate, input_sec)
    cclip = CenterCrop(sampling_rate, input_sec)
    
    def audio_to_data(df, phase):
        gun_file_list = list(df["name"])
        gun_name_list = list(df["cate"]) 
        
        sound_list = []
        label_list = []
        for i in range(len(gun_file_list)):
            sound_name = gun_file_list[i].replace(".mp3", ".wav")
            sound, sr = librosa.load(sound_path + sound_name, mono=True, sr=sampling_rate)
            
            assert sampling_rate == sr
            
            if phase == "train":
                cropped_sound = rclip(sound)
            else:
                cropped_sound = cclip(sound)
                
            sound_list.append(cropped_sound)
            label = name2type[gun_name_list[i]]
            label = le.transform([label])[0]
            label_list.append(label)
            
        sound_array = np.array(sound_list)
        label_array = np.array(label_list)
        
        return sound_array, label_array
            
    x_train, y_train = audio_to_data(train_df, "train")
    x_test, y_test = audio_to_data(test_df, "test")
    
    return x_train, y_train, x_test, y_test, sampling_rate, le.classes_
  
class RandomClip:
    def __init__(self, sampling_rate, clip_length):
        self.clip_length = clip_length * sampling_rate

    def __call__(self, audio_data):
        audio_length = len(audio_data)

        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length-self.clip_length)
            audio_data = audio_data[offset:(offset+self.clip_length)]

        return audio_data

class CenterCrop:
    def __init__(self, sampling_rate, clip_length):
        self.clip_length = clip_length * sampling_rate

    def __call__(self, audio_data):
        audio_length = len(audio_data)

        if audio_length > self.clip_length:
            offset = audio_length //2 - self.clip_length//2
            audio_data = audio_data[offset:(offset+self.clip_length)]

        return audio_data
    
def Fighting_dataloader(filepath=None, debugging=False, useless_data=[]):
    if filepath is None:
        filepath = os.path.join('.', 'dataset', 'Fighting')
    useless_data = [data.lower() for data in useless_data]
        
    sampling_rate = 44100
    label_name = os.listdir(filepath)
    sound_path = []
    label_list = []
    for label in label_name:
        folder_name = os.listdir(os.path.join(filepath, label))
        for folder in folder_name:
            if folder.lower() in useless_data:
                print("{}-{} not used".format(label, folder))
                continue
            file_name = os.listdir(os.path.join(filepath, label, folder))
            for sound_name in file_name:
                sound_file_path = os.path.join(filepath, label, folder, sound_name)
                sound_path.append(sound_file_path)
                label_list.append(label)

    shuffle_index = np.arange(len(sound_path))
    np.random.seed(42)
    np.random.shuffle(shuffle_index)
    train_ratio = int(len(shuffle_index)*0.7)
    valid_ratio = int(len(shuffle_index)*0.85)
    train_index = shuffle_index[:train_ratio]
    valid_index = shuffle_index[train_ratio:valid_ratio]
    test_index = shuffle_index[valid_ratio:]
    le = LabelEncoder()
    le.fit(label_list)

    def audio_to_data(index):
        x_data, y_data = [], []
        for i in index:
            sound, sr = torchaudio.load(sound_path[i])
            print(sound.shape, sound_path[i])
            # if sound.shape[0] == 2:
            #     sound = torch.mean(sound, dim=0).view(1, -1)
            # print(sound_path[i])
            sound = torchaudio.functional.resample(sound, orig_freq=sr, new_freq=sampling_rate)
            x_data.append(sound[0].numpy())
            label = le.transform([label_list[i]])[0]
            y_data.append(label)
        return x_data, y_data
        
    x_train, y_train = audio_to_data(train_index)
    x_val, y_val = audio_to_data(valid_index)
    x_test, y_test = audio_to_data(test_index)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, le.classes_

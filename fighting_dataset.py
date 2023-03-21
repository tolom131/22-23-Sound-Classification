from dataloader import Fighting_dataloader
from utility import * 
from collections import Counter
import os
import numpy as np
import scipy.io as sio

###### embed matlab function into python codes
'''
matlab 사용법
python은 3.8 이상 (중요)
관리자 명령어로 anaconda 실행
C:\Program Files\MATLAB\R2022a\extern\engines\python 로 이동.
python -m pip install . 실행
pip install matlab

'''
def load_fighting_dataset_with_matlab(hop_legnth=20, data_aug=True, debugging=True, useless_data=[]):
    import matlab.engine
    if data_aug:
        if not os.path.exists('data/Fighting/augmentation/'):
            print("First original sound data are loaded.")
            if not os.path.exists('data/Fighting/original'):
                print("You don't have original data.")
                raise FileNotFoundError
            else:
                x_train = np.load('data/Fighting/original/x_train.npy')
                y_train = np.load('data/Fighting/original/y_train.npy') 
                x_val = np.load('data/Fighting/original/x_val.npy')
                y_val = np.load('data/Fighting/original/y_val.npy')
                x_test = np.load('data/Fighting/original/x_test.npy')
                y_test = np.load('data/Fighting/original/y_test.npy')
                train_type = np.load('data/Fighting/train_type.npy')
                
                n_train = x_train.shape[0]
                n_val = x_val.shape[0]
                n_test = x_test.shape[0]
                assert x_train.shape[1] == x_val.shape[1] and x_val.shape[1] == x_test.shape[1]
                sound_length = x_train.shape[1]
                
                max_augment_count = 9
                sampling_rate = 44100
                eng = matlab.engine.start_matlab()
                
                ## x_train augmentation
                print('x_train augmentation, the number of x_train data is {}'.format(n_train))
                ret = eng.AudioDataAugmentation(x_train.tolist(), n_train, sound_length, max_augment_count, sampling_rate)
                aug_x_train = np.array(ret.noncomplex().toarray(), 'float').reshape(-1, sound_length)      
                aug_y_train = np.repeat(y_train, 10)
                # x_val augmentation
                print('x_val augmentation, the number of x_val data is {}'.format(n_val))
                ret = eng.AudioDataAugmentation(x_val.tolist(), n_val, sound_length, max_augment_count, sampling_rate)
                aug_x_val = np.array(ret.noncomplex().toarray(), 'float').reshape(-1, sound_length)      
                aug_y_val = np.repeat(y_val, 10)
                # x_test augmentation
                print('x_test augmentation, the number of x_test data is {}'.format(n_test))
                ret = eng.AudioDataAugmentation(x_test.tolist(), n_test, sound_length, max_augment_count, sampling_rate)
                aug_x_test = np.array(ret.noncomplex().toarray(), 'float').reshape(-1, sound_length)      
                aug_y_test = np.repeat(y_test, 10)
                
                print('save loaded sound data')
                os.makedirs('data/Fighting/augmentation/')
                np.save('data/Fighting/augmentation/x_train.npy', x_train)
                np.save('data/Fighting/augmentation/y_train.npy', y_train)
                np.save('data/Fighting/augmentation/x_val.npy', x_val)
                np.save('data/Fighting/augmentation/y_val.npy', y_val)
                np.save('data/Fighting/augmentation/x_test.npy', x_test)
                np.save('data/Fighting/augmentation/y_test.npy', y_test)                
                print('done')
                
                return aug_x_train, aug_y_train, aug_x_val, aug_y_val, aug_x_test, aug_y_test, sampling_rate, train_type               
        else: 
            x_train = np.load('data/Fighting/augmentation/x_train.npy')
            y_train = np.load('data/Fighting/augmentation/y_train.npy') 
            x_val = np.load('data/Fighting/augmentation/x_val.npy') 
            y_val = np.load('data/Fighting/augmentation/y_val.npy') 
            x_test = np.load('data/Fighting/augmentation/x_test.npy') 
            y_test = np.load('data/Fighting/augmentation/y_test.npy')
            train_type = np.load('data/Fighting/train_type.npy')
            sampling_rate = 44100
            
            return x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type
    else:
        if not os.path.exists('data/Fighting/'):
            print("Create Fighting data.")
            x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type = Fighting_dataloader(debugging=debugging, useless_data=useless_data)
            print("Complete.")

            print("Sound to Feature")
            x_train, y_train = segment_sound(x_train, y_train, sampling_rate, hop_length=hop_legnth)
            x_val, y_val = segment_sound(x_val, y_val, sampling_rate, hop_length=hop_legnth)
            x_test, y_test = segment_sound(x_test, y_test, sampling_rate, hop_length=hop_legnth)
            train_type = np.array(train_type)
            mdic = {
                'x_train' : x_train,
                'y_train' : y_train,
            }
            
            print("Save data")
            if not os.path.exists('data/Fighting'):
                os.makedirs('data/Fighting')
                os.makedirs('data/Fighting/original')
                os.makedirs('data/Fighting/matlab')
            np.save('data/Fighting/original/x_train.npy', x_train)
            np.save('data/Fighting/original/y_train.npy', y_train)
            np.save('data/Fighting/original/x_val.npy', x_val)
            np.save('data/Fighting/original/y_val.npy', y_val)
            np.save('data/Fighting/original/x_test.npy', x_test)
            np.save('data/Fighting/original/y_test.npy', y_test)
            np.save('data/Fighting/train_type.npy', train_type)
            sio.savemat("data/Fighting/matlab/original_sound.mat", mdic)
            
            return x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type
        else:
            x_train = np.load('data/Fighting/original/x_train.npy')
            y_train = np.load('data/Fighting/original/y_train.npy') 
            x_val = np.load('data/Fighting/original/x_val.npy') 
            y_val = np.load('data/Fighting/original/y_val.npy') 
            x_test = np.load('data/Fighting/original/x_test.npy') 
            y_test = np.load('data/Fighting/original/y_test.npy')
            train_type = np.load('data/Fighting/train_type.npy')
            
            sampling_rate = 44100  
            return x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type
        
def load_fighting_dataset_without_matlab(hop_legnth=20, data_aug=True, debugging=True, useless_data=[]):
    if data_aug:
        if not os.path.exists('data/Fighting/matlab/'):
            print("You don't have data augmented data.")
            raise FileNotFoundError
        else:
            mat_contents = sio.loadmat('data/Fighting/matlab/augmented_x_train.mat')      
            x_train = mat_contents['aug_dict'][0][0]
            y_train = mat_contents['aug_dict'][0][1].reshape(-1)
            x_val = np.load('data/Fighting/original/x_val.npy') 
            y_val = np.load('data/Fighting/original/y_val.npy') 
            x_test = np.load('data/Fighting/original/x_test.npy') 
            y_test = np.load('data/Fighting/original/y_test.npy')
            sampling_rate = 44100
            train_type = np.load('data/Fighting/train_type.npy')
            
            return x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type
    else:
        if not os.path.exists('data/Fighting/'):
            print("Create Fighting data.")
            x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type = Fighting_dataloader(debugging=debugging, useless_data=useless_data)
            print("Complete.")

            print("Sound to Feature")
            x_train, y_train = segment_sound(x_train, y_train, sampling_rate, hop_length=hop_legnth)
            x_val, y_val = segment_sound(x_val, y_val, sampling_rate, hop_length=hop_legnth)
            x_test, y_test = segment_sound(x_test, y_test, sampling_rate, hop_length=hop_legnth)
            train_type = np.array(train_type)
            mdic = {
                'x_train' : x_train,
                'y_train' : y_train,
            }
            
            print("Save data")
            if not os.path.exists('data/Fighting'):
                os.makedirs('data/Fighting')
                os.makedirs('data/Fighting/original')
                os.makedirs('data/Fighting/matlab')
            np.save('data/Fighting/original/x_train.npy', x_train)
            np.save('data/Fighting/original/y_train.npy', y_train)
            np.save('data/Fighting/original/x_val.npy', x_val)
            np.save('data/Fighting/original/y_val.npy', y_val)
            np.save('data/Fighting/original/x_test.npy', x_test)
            np.save('data/Fighting/original/y_test.npy', y_test)
            np.save('data/Fighting/train_type.npy', train_type)
            sio.savemat("data/Fighting/matlab/original_sound.mat", mdic)
            
            return x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type
        else:
            x_train = np.load('data/Fighting/original/x_train.npy')
            y_train = np.load('data/Fighting/original/y_train.npy') 
            x_val = np.load('data/Fighting/original/x_val.npy') 
            y_val = np.load('data/Fighting/original/y_val.npy') 
            x_test = np.load('data/Fighting/original/x_test.npy') 
            y_test = np.load('data/Fighting/original/y_test.npy')
            train_type = np.load('data/Fighting/train_type.npy')
            
            sampling_rate = 44100  
            return x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type              
        
            
              
            
        
        
        
        
              
            
import torch.nn as nn
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# from model import SoundExtractor, SoundClassifier
from models import *
from fighting_dataset import *
from utility import *

## 0318: 모델쪽에서 train valid 여부를 판별하는 코드를 추가하였음.
## 이는 models.py 에서 specuamgne를 사용하기 위함임

class FightingDataset(Dataset):
    def __init__(self, sound, label):
        self.sound = sound
        self.label = label
        
    def __getitem__(self, index):
        sound = self.sound[index]
        label = self.label[index]
        return sound, label
    
    def __len__(self):
        return len(self.sound)
    
class ResultParameter:
    def __init__(self):
        self.loss, self.acc, self.f1, self.time = [], [], [], []
 
    def update(self, loss, acc, f1, time):
        self.loss.append(loss)
        self.acc.append(acc)
        self.f1.append(f1)
        self.time.append(time)
        
    def update(self, loss, acc, f1):
        self.loss.append(loss)
        self.acc.append(acc)
        self.f1.append(f1)
        
    def get_loss(self): return sum(self.loss) / len(self.loss)
    def get_acc(self): return sum(self.acc) / len(self.acc)   
    def get_f1(self): return sum(self.f1) / len(self.f1)
    def get_time(self): return sum(self.time) / len(self.time)
    
def make_data(use_aug=True, useless_data=[]):
    x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type = load_fighting_dataset_without_matlab(hop_legnth=20, data_aug=use_aug, useless_data=useless_data)
    return x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type
    
def train(use_aug=True, useless_data=[]):
    print('load sound data')
    x_train, y_train, x_val, y_val, x_test, y_test, sampling_rate, train_type = make_data(use_aug, useless_data)

    counter_ = Counter(y_train).items()
    counter_ = sorted(counter_, key=lambda x: x[0])
    print("train: ", counter_)

    counter_ = Counter(y_val).items()
    counter_ = sorted(counter_, key=lambda x: x[0])
    print("valid: ", counter_)

    counter_ = Counter(y_test).items()
    counter_ = sorted(counter_, key=lambda x: x[0])
    print("test: ", counter_)
    print("test: ", train_type)
    
    dataset_train = FightingDataset(x_train, y_train)
    dataset_val = FightingDataset(x_val, y_val)
    dataset_test = FightingDataset(x_test, y_test)
    
    lr = 0.000005
    batch_size = 128
    epochs = 300
    early_stop_maximum = 5
    hdim = 128
    n_mels = 64
    num_layers = 4
    n_classes = len(set(y_train))

    # backbone = SoundExtractor(hdim, sample_rate=sampling_rate, n_fft=x_train.shape[1], n_mels=n_mels).cuda()
    # classifier = SoundClassifier(hdim, n_classes, num_layer=num_layers).cuda()
    backbone = CTransExtractor(hdim, sample_rate=sampling_rate, n_fft=x_train.shape[1], n_mels=n_mels, num_layer=num_layers).cuda()
    classifier = SingleClassifer(hdim, n_classes).cuda()    
    savepath_backbone = 'transformer/backbone.pt'
    savepath_classifier = 'transformer/classifier.pt'

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False, drop_last=True)

    params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    cls_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f'# of Backbone Parameters:{params}, # of Classifier Parameters: {cls_params}')
    
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=0.005)

    best_val_ce = 1000
    early_stop_count = 0
    
    train_loss = []
    train_acc = []
    train_f1 = []

    test_loss = []
    test_acc = []
    test_f1 = []
    test_time = []

    for epoch in range(epochs):
        backbone.train()
        classifier.train()

        ce_loss = 0.0
        cate_preds, cate_trues = [], []
        for step, (wv, cate) in enumerate(dataloader_train):
            wv = wv.float()
            cate = cate.type(torch.LongTensor)
            wv, cate = wv.cuda(), cate.cuda()

            features = backbone(wv, 'train')
            cate_out = classifier(features)
            cate_loss = ce(cate_out, cate)
            cate_pred = torch.argmax(cate_out, dim=-1)

            loss = cate_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ce_loss += loss.item()

            cate_preds.append(cate_pred.detach().cpu())
            cate_trues.append(cate.detach().cpu())

        cate_preds = torch.cat(cate_preds, dim=0).numpy()
        cate_trues = torch.cat(cate_trues, dim=0).numpy()

        train_ce_loss = ce_loss / len(dataloader_train)
        train_cate_acc = accuracy_score(cate_trues, cate_preds)
        train_cate_f1 = f1_score(cate_trues, cate_preds, average='macro')
        
        train_loss.append(train_ce_loss)
        train_acc.append(train_cate_acc)
        train_f1.append(train_cate_f1)

        with torch.no_grad():
            backbone.eval()
            classifier.eval()

            ce_loss = 0.0
            cate_preds, cate_trues = [], []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            timings=np.zeros((len(dataloader_val), 1))
            for step, (wv, cate) in enumerate(dataloader_val):
                wv = wv.float()
                cate = cate.type(torch.LongTensor)
                wv, cate = wv.cuda(), cate.cuda()

                start.record()
                features = backbone(wv, 'valid')
                cate_out = classifier(features)
                end.record()
                cate_loss = ce(cate_out, cate)
                
                torch.cuda.synchronize()
                curr_time = start.elapsed_time(end)
                timings[step] = curr_time
                    
                cate_pred = torch.argmax(cate_out, dim=-1)
                cate_preds.append(cate_pred.cpu())
                cate_trues.append(cate.cpu())
                ce_loss += cate_loss.item()

            cate_preds = torch.cat(cate_preds, dim=0).numpy()               
            cate_trues = torch.cat(cate_trues, dim=0).numpy()

            test_ce_loss = ce_loss / len(dataloader_val)
            test_cate_acc = accuracy_score(cate_trues, cate_preds)
            test_cate_f1 = f1_score(cate_trues, cate_preds, average='macro')
            test_cate_time = np.mean(timings)
            
            test_loss.append(test_ce_loss)
            test_acc.append(test_cate_acc)
            test_f1.append(test_cate_f1)
            test_time.append(test_cate_time)

        print(epoch, f'train ce:{train_ce_loss:.4f}, train cate-acc:{train_cate_acc:.4f}, ' f'train cate-F1:{train_cate_f1:.4f}')
        print("\t", f' valid ce:{test_ce_loss:.4f}, valid cate-acc:{test_cate_acc:.4f},' f'valid cate-F1:{test_cate_f1:.4f}', f'valid time:{test_cate_time:.4f}')
        print(f'best valid ce: {best_val_ce}')
        if best_val_ce >= test_ce_loss:
            best_val_ce = test_ce_loss
            print(f'backbone saved at {savepath_backbone}')
            save_model(backbone, savepath_backbone)
            save_model(classifier, savepath_classifier)
            early_stop_count = 0
        else:
            early_stop_count += 1
            print("early_stop_count: ", early_stop_count)
            if early_stop_count == early_stop_maximum:
                print("early stop.")
                break
        print("")    
    backbone.load_state_dict(torch.load(savepath_backbone))
    classifier.load_state_dict(torch.load(savepath_classifier))

    for a, b in dataloader_test:
        a = a.float()
        a = a.cuda()
        y = backbone(a, 'test')
        y_pred = classifier(y)
        
    y_pred = y_pred.cpu()
    acc = accuracy_score(y_pred.argmax(axis=1), y_test)
    score = f1_score(y_test, y_pred.argmax(axis=1), average="macro")
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1), normalize='true')
    print("accuracy: ", acc)
    print("f1 score : ", score)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation="nearest", cmap="OrRd")
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    fig.colorbar(cax)
    plt.title("Fighting VR genre model result")
    plt.ylabel("true labels")
    plt.xlabel("predicted labels")
    ax.set_xticklabels(['']+list(train_type))
    ax.set_yticklabels(['']+list(train_type))
    plt.savefig('train result.png')

if __name__ == "__main__":
    use_aug = True
    # useless_data = ['Artlist', 'Freesound']
    useless_data = []
    # make_data(use_aug, useless_data)
    train(use_aug, useless_data)
    
    
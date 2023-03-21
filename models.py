import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np

from transformer import PositionalEncoding, TransformerEncoderBlock
from augment import SpecAugment

# model.py 만들기 전 버전
# 여기에 SpecAugment 추가하였음.

class OurExtractor(nn.Module):
    def __init__(self, hidden_dim,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96):
        """
        Args:
          sample_rate (int): path to load dataset from
          n_fft (int): number of samples for fft
          f_min (float): min freq
          f_max (float): max freq
          n_mels (float): number of mel bin
          n_class (int): number of class
        """
        super(OurExtractor, self).__init__()
        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(2)

        self.conv0 = nn.Sequential(
            nn.Conv1d(n_mels*2, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )


    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        B, C, M, T = x.shape
        x = x.reshape(B, C*M, T) # for 1D conv
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class CTransExtractor(nn.Module):
    def __init__(self, hidden_dim,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96,
                 num_layer=1):
        """
        Args:
          sample_rate (int): path to load dataset from
          n_fft (int): number of samples for fft
          f_min (float): min freq
          f_max (float): max freq
          n_mels (float): number of mel bin
          n_class (int): number of class
        """
        super(CTransExtractor, self).__init__()
        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels,
                                                         center = True)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm1d(n_mels)

        self.conv0 = nn.Sequential(
            nn.Conv1d(n_mels, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        # Predict tag using the aggregated features.

        self.position_encoder = PositionalEncoding(hidden_dim, 25)
        self.dropout = nn.Dropout(0.5)
        self.num_layers = num_layer

        encoders = [TransformerEncoderBlock(hidden_dim=hidden_dim, dropout=0.5, n_head=4, feed_forward_dim=hidden_dim) \
                    for _ in range(self.num_layers)]
        self.encoders = nn.ModuleList(encoders)
        
    def forward(self, x, phase):
        x = self.spec(x)
        x = self.to_db(x)
        # x = self.spec_bn(x)                 
        B, T, C = x.shape
        x = x.contiguous()
        
        if phase == 'train':
            x = np.zeros(shape=(B, T, C))
            for i in range(B):
                x_ = x[i]
                apply = SpecAugment(x_, policy='SC')
                x_ = apply.freq_mask()
                x_ = apply.time_mask()
                x[i] = x_   
        x = self.conv0(apply.mel_spectrogram)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.permute(2, 0, 1).contiguous()
        sl, _, _ = x.shape
        padding_mask = torch.zeros(sl, B).bool().cuda()
        out = self.position_encoder(x)
        out = self.dropout(out)
        for encoder in self.encoders:
            out = encoder(out, padding_mask=padding_mask)
        # L, B, C
        return out.mean(dim=0)

class SingleClassifer(nn.Module):
    def __init__(self, hidden_dim, n_class):
        super(SingleClassifer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, n_class))

    def forward(self, feature):
        logit = self.fc(feature.flatten(start_dim=1))
        return logit

class OurClassifer(nn.Module):
    def __init__(self, hidden_dim, n_cate):
        super(OurClassifer, self).__init__()

        # gun types: Transformer
        self.position_encoder = PositionalEncoding(hidden_dim, 25)
        self.dropout = nn.Dropout(0.5)
        self.num_layers = 1

        encoders = [TransformerEncoderBlock(hidden_dim=hidden_dim, dropout=0.5, n_head=4, feed_forward_dim=hidden_dim) \
                    for _ in range(self.num_layers)]
        self.encoders = nn.ModuleList(encoders)
        self.cate_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_cate))

    def forward(self, feature):

        B, _, _ = feature.shape
        x = feature.clone().permute(2, 0, 1).contiguous()
        padding_mask = torch.zeros(21, B).bool().cuda()
        out = self.position_encoder(x)
        out = self.dropout(out)
        for encoder in self.encoders:
            out = encoder(out, padding_mask=padding_mask)
        # L, B, C
        cate_logit = self.cate_fc(out.mean(dim=0))
        
        return cate_logit


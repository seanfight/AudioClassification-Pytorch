import torch
from torch import nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt
waveform, sample_rate = torchaudio.load(r'dataset/UrbanSound8K/audio/fold2/204773-3-8-0.wav', normalize=True)

transform = torchaudio.transforms.MelSpectrogram(sample_rate)
mel_specgram = transform(waveform)  # (channel, n_mels, time)
print(mel_specgram.shape)
print(type(mel_specgram))

# print(mel_specgram.transpose(2,1).shape)
# mean = torch.mean(mel_specgram.transpose(2,1), 1, keepdim=True)

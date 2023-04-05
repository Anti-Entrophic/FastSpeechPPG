import random
import numpy as np
import torch
import torch.utils.data
# from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from scipy.io.wavfile import read

import layers
from debugging import logging_init
import logging

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_PPG(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_PPG = [line.strip().split(split) for line in f]
    return filepaths_and_PPG
  
def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
    
class PPG_MelLoader_test(torch.utils.data.Dataset):
    """
        test phoneme_train_corpus.csv
    """
    def __init__(self, audiopaths_and_PPG, hparams):
        logging_init()
        # 这个时候audiopaths_and_PPG是一个列表套列表[[audiopaths,PPGpath],[audiopaths,PPGpath]...]
        # 其中audiopaths是音频路径，PPGpath是存储PPG的npy文件
        self.audiopaths_and_PPG = load_filepaths_and_PPG(audiopaths_and_PPG)
        
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        # shuffle一下，让原来按顺序读入变成乱序
        random.shuffle(self.audiopaths_and_PPG)

    def get_mel_PPG_pair(self, audiopath_and_PPG):
        # separate audiopath and PPG
        # 这里后面的不是PPG了，是一长串列表里面是音素。
        audiopath, PPG = audiopath_and_PPG[0], audiopath_and_PPG[1]
        
        # print(type(PPG)) 是str
        
        # 首先我要先建立一个dict，对应音素和序号。
        pho_map = self.create_map("/content/FastSpeechPPG/FastSpeech2/filelists/phoneme.characters")
        
        # 先获取PPG和speaker_embedding
        # speaker_embedding = self.get_id(audiopath)
        PPG = self.get_ppg(PPG, pho_map)
        mel = self.get_mel(audiopath)
        mel = mel.permute(1, 0)
        if(len(PPG)!=len(mel)):
          if(len(PPG)>len(mel)):
            PPG = PPG[0:len(mel)]
          else:
            mel = mel[0:len(PPG)]
        mel = mel.permute(1, 0)
        return (PPG, mel)

    def create_map(self, filepath):
        # 建立音素到序号的映射
        with open(filepath, encoding='utf-8') as f:
            it = 0
            pho_name = {}
            for line in f :
                pho_name[line.strip()]  = it
                it = it + 1
        return pho_name
    
    def get_id(self, audiopath):
        audiopath.split('/')
        speaker_embedding = np.load(audiopath[4]+'.npy')
        return speaker_embedding
    
    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_ppg(self, PPG, pho_map):
        # 传入的是一个包含音素，由空格分隔的字符串
        # 分割完得到一个列表
        
        pho_id_list = [pho_map[pho] for pho in PPG.split()]

        # 这里的PPG就是对应的ont-hot向量
        PPG_temp = np.eye(72)[pho_id_list]
        PPG_temp = torch.from_numpy(PPG_temp)
        # 应该还要动model.py里的Tacotron2 Class. 原本TextMelLoader只是传出text的sequence, 之后是在Tacotron2 Class里每个embedding成512维
        # for frame in PPG_temp:
        #     frame = np.append(frame, speaker_embedding)
        return PPG_temp

    def __getitem__(self, index):
        # 等于是按行读入
        return self.get_mel_PPG_pair(self.audiopaths_and_PPG[index])

    def __len__(self):
        return len(self.audiopaths_and_PPG)

class PPGMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from PPG and mel-spectrogram
        PARAMS
        ------
        batch: [PPG, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        PPG_padded = torch.LongTensor(len(batch), max_input_len, 72) # PPG.size(1)是音素个数

        PPG_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            PPG = batch[ids_sorted_decreasing[i]][0]
            PPG_padded[i, :PPG.size(0)] = PPG
        # logging.debug(PPG_padded)

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return PPG_padded, input_lengths, mel_padded, gate_padded, output_lengths

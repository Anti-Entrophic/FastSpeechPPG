import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
import logging


def prepare_align(config):

    logging.basicConfig(level=logging.DEBUG #设置日志输出格式
      ,filename="/content/experiment.log" #log日志输出的文件位置和文件名
      ,format="%(asctime)s-%(levelname)s: %(message)s" #日志输出的格式
                      # -8表示占位符，让输出左对齐，输出长度都为8位
      ,datefmt="%Y-%m-%d %H:%M:%S"  #时间输出的格式
      ,force=True)

    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "LJSpeech"
    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        for line in tqdm(f):
            # line be like: LJ001-0001|Printing, in the only sense with which...|Printing, in the only sense with which...
            # 我只找到一行part[1]和parts[2]不一样的，不知道啥意思
            parts = line.strip().split("|")

            base_name = parts[0]
            text = parts[2]

            # 没看出来clean了啥，好像就大写变小写。大概还会删掉一些特殊字符啥的吧
            text = _clean_text(text, cleaners)

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                # type(wav) = 'numpy.ndarray'
                wav = wav / max(abs(wav)) * max_wav_value

                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)
    assert 0
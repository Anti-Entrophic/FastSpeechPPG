import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

from debugging import logging_init
import logging

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        logging_init()
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(self, PPG, input_lengths, mels, max_len, mel_lens):
        # PPG, input_lengths, mels, max_len, mel_lens = inputs
        logging.debug("in fastspeech 2, forward")
        # 我不清楚这步.data有什么意义
        input_lengths, mel_lens = input_lengths.data, mel_lens.data
        '''
        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        '''
        output = PPG

        max_mel_len = max(mel_lens)

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        logging.debug("mel_masks")
        logging.debug(mel_masks)
        logging.debug(mel_masks.shape)

        logging.debug("-----through decoder-----")
        output, mel_masks = self.decoder(output, mel_masks)
        logging.debug("decoder output:")
        logging.debug(output)
        logging.debug(output.shape)
        logging.debug("mel_masks:")
        logging.debug(mel_masks)
        logging.debug(mel_masks.shape)

        logging.debug("-----mel linear output-----")
        output = self.mel_linear(output)
        logging.debug(output)
        logging.debug(output.shape)

        logging.debug("-----postnet-----")
        temp = self.postnet(output)
        logging.debug("postnet(output):")
        logging.debug(temp)
        logging.debug(temp.shape)

        postnet_output = temp + output
        logging.debug("postnet_output")
        logging.debug(postnet_output)
        logging.debug(postnet_output.shape)

        return (
            output,
            postnet_output,
            mel_masks,
            mel_lens,
        )

import torch
import torch.nn as nn

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, PPG, input_lengths, mels, max_len, mel_lens, mel_predictions, postnet_mel_predictions, mel_masks):
        ''' predictions:
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        '''
        #(mel_predictions, postnet_mel_predictions, mel_masks) = predictions
        
        # src_masks = ~src_masks
        mel_masks = ~mel_masks
        # log_duration_targets = torch.log(duration_targets.float() + 1)

        mel_targets = mels[:, : mel_masks.shape[1], :]

        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        # log_duration_targets.requires_grad = False
        # pitch_targets.requires_grad = False
        # energy_targets.requires_grad = False
        # mel_targets.requires_grad = False
        '''
        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)
        
        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        '''
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )

        mel_targets = mel_targets.permute(0, 2, 1)

        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # debug到这一行
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        # pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        # energy_loss = self.mse_loss(energy_predictions, energy_targets)
        # duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
        )

import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from data_utils import PPG_MelLoader_test, PPGMelCollate, to_gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, hparams, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs
    '''
    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    '''
    collate_fnn = PPGMelCollate(hparams.n_frames_per_step)
    dataset = PPG_MelLoader_test(hparams.training_files, hparams)
    batch_size = train_config["optimizer"]["batch_size"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fnn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    for batchs in loader:
        PPG = batchs[0]
        input_lengths = batchs[1] 
        mel_padded = batchs[2]
        gate_padded = batchs[3]
        output_lengths = batchs[4]

        PPG = to_gpu(PPG).float()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        # Forward
        output = model(PPG, input_lengths, mel_padded, max_len, output_lengths)
        mel_predictions = output[0]
        postnet_mel_predictions = output[1]
        mel_masks = output[2]
                
        # Cal Loss
        losses = Loss(PPG, input_lengths, mel_padded, max_len, output_lengths, 
            mel_predictions, postnet_mel_predictions, mel_masks)
        total_loss = losses[0]

        for i in range(len(losses)):
            loss_sums[i] += losses[i].item() * len(batchs[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )
    '''
    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batchs,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )
    '''
    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)

import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from data_utils import PPG_MelLoader_test, PPGMelCollate, to_gpu

from evaluate import evaluate

from debugging import logging_init
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class create_hparams():
    """Create model hyperparameters. Parse nondefault from given string."""
    ################################
    #       CUDA Enable            #
    ################################
    if torch.cuda.is_available() :
        cuda_enabled = True    
    else :
        cuda_enabled = False

    ################################
    # Experiment Parameters        #
    ################################
    epochs = 100
    iters_per_checkpoint = 500
    seed= 1234
    dynamic_loss_scaling = True
    fp16_run = False
    distributed_run = False
    dist_backend = "nccl"
    dist_url = "tcp://localhost:54321"
    cudnn_enabled = True
    cudnn_benchmark = False
    ignore_layers = ['embedding.weight']

    ################################
    # Data Parameters             #
    ################################
    load_mel_from_disk = False
    training_files = 'filelists/phoneme_test_wav.tsv'
    validation_files = 'filelists/phoneme_test_wav.tsv'
    text_cleaners = ['japanese_cleaners']

    ################################
    # Audio Parameters             #
    ################################
    max_wav_value = 32768.0
    sampling_rate = 16000
    filter_length = 1024
    hop_length = 160
    win_length = 400
    n_mel_channels = 80
    mel_fmin = 0.0
    mel_fmax = 8000.0 
    # Decoder parameters
    n_frames_per_step = 1  # currently only 1 is supported

def main(args, configs):
    logging_init()
    print("Prepare training ...")
    logging.debug("-----------------------")
    logging.debug("-----------------------")
    logging.debug("-----------------------")
    hparams = create_hparams()
    preprocess_config, model_config, train_config = configs

    # Get dataset
    '''
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    '''
    collate_fnn = PPGMelCollate(hparams.n_frames_per_step)
    dataset = PPG_MelLoader_test(hparams.training_files, hparams)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset

    loader = DataLoader(
        dataset,
        #batch_size=batch_size * group_size,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fnn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
                # PPG, input_lengths, mel_padded, gate_padded, output_lengths = batchs

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
                
                logging.debug("PPG:")
                logging.debug(PPG)
                logging.debug(PPG.shape)
                logging.debug(PPG[0].shape)
                logging.debug("input_lengths:")
                logging.debug(input_lengths)
                logging.debug(input_lengths.shape)
                logging.debug("mel_padded:")
                logging.debug(mel_padded)
                logging.debug(mel_padded.shape)
                logging.debug(mel_padded[0].shape)
                logging.debug("gate_padded:")
                logging.debug(gate_padded)
                logging.debug(gate_padded.shape)
                logging.debug("output_lengths:")
                logging.debug(output_lengths)
                logging.debug(output_lengths.shape)
                
                
                # Forward
                output = model(PPG, input_lengths, mel_padded, max_len, output_lengths)
                mel_predictions = output[0]
                postnet_mel_predictions = output[1]
                mel_masks = output[2]
                
                # Cal Loss
                losses = Loss(PPG, input_lengths, mel_padded, max_len, output_lengths, 
                    mel_predictions, postnet_mel_predictions, mel_masks)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batchs,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

                inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
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

    main(args, configs)

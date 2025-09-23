import os.path

import librosa
import numpy as np
import tqdm
from dotmap import DotMap
from torch.utils.data import DataLoader
from model import CNN_DNN
from utils.dataloader import GRIDDataset
from torch.optim import AdamW
import torch.nn as nn
import torch
from pystoi import stoi


class GESLoss:
  def __init__(self, a_imag=2.5, a_ph=0.1):
    self.a_imag = a_imag
    self.a_ph = a_ph

  def __call__(self, cirm_r, cirm_i, cirm_r_hat, cirm_i_hat, phase_corr):
    # Compute predicted phase angle
    phase_corr_hat = torch.atan2(cirm_i_hat, cirm_r_hat)

    # Compute weighted loss
    loss = 0.5 * torch.mean(
      torch.sum((cirm_r - cirm_r_hat) ** 2, dim=-1)
      + self.a_imag * torch.sum((cirm_i - cirm_i_hat) ** 2, dim=-1)
      + self.a_ph * torch.sum(torch.abs(phase_corr - phase_corr_hat), dim=-1)
    )
    return loss


def produce_enhanced_signal(cirm_r_hat, cirm_i_hat):
  # Invert sigmoid function scaling
  cirm_r_hat_inv = torch.log(cirm_r_hat / (1 - cirm_r_hat))
  cirm_i_hat_inv = torch.log(cirm_i_hat / (1 - cirm_i_hat))


def train_loop(model, optimiser, loss_fn, train_loader, epoch):
  epoch_loss = 0

  for log_power, cirm_r, cirm_i, phase_corr, _ in tqdm.tqdm(train_loader, desc=f"Training epoch {epoch + 1}"):
    log_power = log_power.cuda()
    cirm_r = cirm_r.cuda()
    cirm_i = cirm_i.cuda()
    phase_corr = phase_corr.cuda()

    optimiser.zero_grad()
    cirm_r_hat, cirm_i_hat = model(log_power)
    loss = loss_fn(cirm_r, cirm_i, cirm_r_hat, cirm_i_hat, phase_corr)
    loss.backward()
    optimiser.step()

    epoch_loss += loss.item()

  return epoch_loss / len(train_loader)


def validate(model, val_loader, epoch, args):
  model.eval()
  stoi_scores = []

  # Ensure we get the same set each time
  val_loader.dataset.reset_seed()

  for log_power, clean_audio, noisy_stft in tqdm.tqdm(val_loader, desc=f"Validating epoch {epoch + 1}"):
    with torch.no_grad():
      t_max = log_power.shape[-1]
      t_end_start = t_max - args.n_frames + 1
      real_mask_hat = torch.zeros_like(log_power).squeeze(1)
      imag_mask_hat = torch.zeros_like(log_power).squeeze(1)

      # Populate the real and imaginary masks at the center of each window
      for t in range(t_end_start):
        frame_batch = log_power[..., t:t+args.n_frames]
        frame_batch = frame_batch.cuda()
        cirm_r_hat, cirm_i_hat = model(frame_batch)
        center_idx = t + (args.n_frames // 2)
        real_mask_hat[..., center_idx] = cirm_r_hat
        imag_mask_hat[..., center_idx] = cirm_i_hat

      # Multiply STFT by predicted real and imaginary masks
      mask_hat = torch.complex(real_mask_hat, imag_mask_hat)
      enhanced_stft = mask_hat * noisy_stft

      # Compute audio length
      n_fft = int(args.window_size * args.target_sr)
      hop_length = int(n_fft * args.window_overlap)
      target_len = n_fft + (args.n_frames_test - 1) * hop_length

      # Inverse enhanced STFT to audio
      window = np.hanning(n_fft)
      enhanced_stft_np = enhanced_stft.detach().cpu().numpy()
      enhanced_audio = librosa.istft(
        enhanced_stft_np,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        length=target_len
      )
      for clean, enhanced in zip(clean_audio, enhanced_audio):
        score = stoi(clean, enhanced, args.target_sr, extended=True)
        stoi_scores.append(score)

  return np.mean(np.array(stoi_scores))


def train(args):
  train_dataset = GRIDDataset(dataset_dir="D:/Datasets/GRID/Splits", noise_dir="D:/Datasets/DEMAND/Splits",
                              split="train", args=args, seed=0, test=False)
  val_dataset = GRIDDataset(dataset_dir="D:/Datasets/GRID/Splits", noise_dir="D:/Datasets/DEMAND/Splits",
                              split="val", args=args, seed=1, test=True)
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4)
  val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)

  # Load CNN+DNN model
  model = CNN_DNN().cuda()
  model.train()

  # Optimiser and loss functions
  optimiser = AdamW(model.parameters())
  loss_fn = GESLoss(a_imag=2.5, a_ph=0.1)

  # Weights directory
  os.makedirs(args.weights_dir, exist_ok=True)

  # Training iterations
  best_val_estoi = 0.0
  for epoch in range(args.epochs):
    train_loss = train_loop(model, optimiser, loss_fn, train_loader, epoch)
    val_estoi = validate(model, val_loader, epoch, args)
    print(f"Train loss: {train_loss} | Validation ESTOI: {val_estoi}")
    if val_estoi > best_val_estoi:
      best_val_estoi = val_estoi
      print(f"Best weights saved (Validation ESTOI: {val_estoi})")
      torch.save(model.state_dict(), os.path.join(args.weights_dir, "best_weights.pth"))


if __name__ == "__main__":
  args = DotMap()
  args.batch_size = 64
  args.subset = 0.2
  args.target_sr = 16000
  args.window_size = 0.02
  args.window_overlap = 0.5
  args.n_frames = 47
  args.n_frames_test = 299
  args.snr = 0
  args.epochs = 100
  args.audio_ext = ".wav"
  args.noise_ext = ".wav"
  args.weights_dir = "weights"

  train(args)

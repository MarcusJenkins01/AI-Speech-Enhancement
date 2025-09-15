import tqdm
from dotmap import DotMap
from torch.utils.data import DataLoader
from model import CNN_DNN
from utils.dataloader import GRIDDataset
from torch.optim import AdamW
import torch.nn as nn
import torch


class GESLoss:
  def __init__(self, a_imag=2.5, a_ph=0.1):
    self.a_imag = a_imag
    self.a_ph = a_ph

  def __call__(self, cirm_r, cirm_i, cirm_r_hat, cirm_i_hat, phase_corr):
    # Compute predicted phase correction
    phase_corr_hat = torch.atan2(cirm_i_hat, cirm_r_hat)

    # Compute weighted loss
    loss = 0.5 * torch.mean(
      torch.sum((cirm_r - cirm_r_hat) ** 2, dim=-1)
      + self.a_imag * torch.sum((cirm_i - cirm_i_hat) ** 2, dim=-1)
      + self.a_ph * torch.sum((phase_corr - phase_corr_hat) ** 2, dim=-1)
    )
    return loss


def train(args):
  train_dataset = GRIDDataset(dataset_dir="D:/Datasets/GRID/Splits", noise_dir="D:/Datasets/DEMAND/Splits",
                              split="train", snr=args.snr, target_sr=args.target_sr, n_frames=args.n_frames)
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)

  model = CNN_DNN().cuda()
  model.train()

  optimiser = AdamW(model.parameters())
  loss_fn = GESLoss(a_imag=2.5, a_ph=0.1)

  for epoch in range(args.epochs):
    epoch_loss = 0

    for log_power, cirm_r, cirm_i, phase_corr in tqdm.tqdm(train_loader, desc=f"Training epoch {epoch + 1}"):
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

    print(f"Mean loss: {epoch_loss / len(train_loader)}")

if __name__ == "__main__":
  args = DotMap()
  args.target_sr = 16000
  args.n_frames = 47
  args.snr = 0
  args.epochs = 100

  train(args)

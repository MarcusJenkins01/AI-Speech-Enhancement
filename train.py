from dotmap import DotMap
from torch.utils.data import DataLoader
from model import CNN_DNN
from utils.dataloader import GRIDDataset


def train(args):
  train_dataset = GRIDDataset(dataset_dir="D:/Datasets/GRID/Splits", noise_dir="D:/Datasets/DEMAND/Splits",
                              split="train", snr=args.snr, target_sr=args.target_sr, n_frames=args.n_frames)
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)

  model = CNN_DNN()

  for log_power, mask_real, mask_imag, phase_corr in train_loader:
    real, imag = model(log_power)
    print(real.shape, imag.shape)


if __name__ == "__main__":
  args = DotMap()
  args.target_sr = 16000
  args.n_frames = 47
  args.snr = 0

  train(args)

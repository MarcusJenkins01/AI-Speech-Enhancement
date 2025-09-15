import math
import numpy as np
import sounddevice
from torch.utils.data import Dataset, DataLoader
import glob
import os
import random
import librosa


class GRIDDataset(Dataset):
  def __init__(self, dataset_dir, noise_dir, split, snr=0, audio_ext=".wav", noise_ext=".wav", seed=0,
               target_sr=16000, window_size=0.02, window_overlap=0.5, n_frames=47):
    super().__init__()
    assert "." in audio_ext and "." in noise_ext, "Audio extensions must be .<ext>"
    self.split = split
    self.snr = snr
    self.audio_paths = glob.glob(os.path.join(dataset_dir, split, f"*/*{audio_ext}"))
    self.noise_paths = glob.glob(os.path.join(noise_dir, split, f"*{noise_ext}"))
    self.noise_path_rng = random.Random(seed)
    self.noise_crop_rng = random.Random(seed)
    self.target_sr = target_sr
    self.target_length = target_length
    self.window_size = window_size
    self.window_overlap = window_overlap
    self.n_frames = n_frames
    self.n_fft = int(self.window_size * self.target_sr)
    self.hop_length = int(self.n_fft * self.window_overlap)
    self.target_length = self.n_fft + (self.n_frames - 1) * self.hop_length

  def __len__(self):
    return len(self.audio_paths)

  def _load_resample(self, audio_path):
    audio_signal, a_sr = librosa.load(audio_path, sr=None)
    if a_sr != self.target_sr:
      audio_signal = librosa.resample(audio_signal, orig_sr=a_sr, target_sr=self.target_sr)
    return audio_signal

  def _align_noise_length(self, noise_signal):
    if len(noise_signal) == self.target_length:
      return noise_signal

    if len(noise_signal) < self.target_length:
      repeats = math.ceil(self.target_length / len(noise_signal))
      noise_signal = np.tile(noise_signal, repeats)

    start = self.noise_crop_rng.randint(0, len(noise_signal) - self.target_length + 1)
    return noise_signal[start:start + self.target_length]

  def _mix_with_noise(self, audio_signal, noise_signal):
    # Select a random crop of the noise based on target_length
    noise_segment = self._align_noise_length(noise_signal)

    # Scale noise based on desired SNR
    audio_power = np.mean(audio_signal ** 2)
    noise_power = np.mean(noise_segment ** 2)
    snr_linear = 10 ** (self.snr / 10)
    scaling_factor = np.sqrt(audio_power / (noise_power * snr_linear))
    noise_segment = noise_segment * scaling_factor

    # Mix audio into the noise
    if len(audio_signal) > self.target_length:
      # Randomly choose a crop of the speech
      start = self.noise_crop_rng.randint(0, len(audio_signal) - self.target_length)
      audio_signal = audio_signal[start:start + self.target_length]
      audio_offset = 0
    else:
      # Random offset if audio is shorter than the target length
      audio_offset = self.noise_crop_rng.randint(0, self.target_length - len(audio_signal) + 1)

    mixed_audio = noise_segment.copy()
    mixed_audio[audio_offset:audio_offset + len(audio_signal)] += audio_signal

    clean_padded = np.zeros_like(noise_segment)
    clean_padded[audio_offset:audio_offset + len(audio_signal)] = audio_signal

    return mixed_audio, clean_padded

  def _get_targets(self, mixed_audio, clean_padded, eps=1e-8):
    # Compute STFTs
    han_window = np.hanning(self.n_fft)
    noisy_stft = librosa.stft(mixed_audio, n_fft=self.n_fft, hop_length=self.hop_length, window=han_window, center=True)
    clean_stft = librosa.stft(clean_padded, n_fft=self.n_fft, hop_length=self.hop_length, window=han_window, center=True)

    # Ideal complex ratio mask (CIRM)
    cirm = clean_stft / (noisy_stft + eps)
    mask_real = np.real(cirm)
    mask_imag = np.imag(cirm)

    # Phase correction angle
    phase_noisy = np.angle(noisy_stft)
    phase_clean = np.angle(clean_stft)
    phase_corr = phase_clean - phase_noisy
    phase_corr = (phase_corr + np.pi) % (2 * np.pi) - np.pi

    return mask_real, mask_imag, phase_corr

  def __getitem__(self, idx):
    audio_path = self.audio_paths[idx]
    noise_path = self.noise_paths[self.noise_path_rng.randint(0, len(self.noise_paths) - 1)]

    audio_signal = self._load_resample(audio_path)
    noise_signal = self._load_resample(noise_path)
    mixed_audio, clean_padded = self._mix_with_noise(audio_signal, noise_signal)

    mask_real, mask_imag, phase_corr = self._get_targets(mixed_audio, clean_padded)

    return mixed_audio, clean_padded


if __name__ == "__main__":
  #############
  # Debugging #
  #############
  target_length = 48000
  target_snr = 16000
  train_dataset = GRIDDataset(dataset_dir="D:/Datasets/GRID/Splits", noise_dir="D:/Datasets/DEMAND/Splits",
                             split="train", snr=0, target_sr=target_snr, n_frames=47)
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)

  for mixed_audio, clean_padded in train_loader:
    mixed_audio = mixed_audio[0]
    sounddevice.play(mixed_audio, samplerate=target_snr)
    sounddevice.wait()

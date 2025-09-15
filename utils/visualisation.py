import librosa
import numpy as np
from matplotlib import pyplot as plt


def plot_targets(mask_real, mask_imag, phase_corr, hop_length=160, sr=16000):
  mask_real_clipped = np.clip(mask_real, -2, 2)
  mask_imag_clipped = np.clip(mask_imag, -2, 2)

  fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

  img0 = librosa.display.specshow(mask_real_clipped,
                                  sr=sr,
                                  hop_length=hop_length,
                                  x_axis="time",
                                  y_axis="linear",
                                  ax=ax[0])
  ax[0].set_title("Real part of Ideal Complex Ratio Mask (CIRM)")
  fig.colorbar(img0, ax=ax[0], format="%+0.2f")

  img1 = librosa.display.specshow(mask_imag_clipped,
                                  sr=sr,
                                  hop_length=hop_length,
                                  x_axis="time",
                                  y_axis="linear",
                                  ax=ax[1])
  ax[1].set_title("Imaginary part of Ideal Complex Ratio Mask (CIRM)")
  fig.colorbar(img1, ax=ax[1], format="%+0.2f")

  img2 = librosa.display.specshow(phase_corr,
                                  sr=sr,
                                  hop_length=hop_length,
                                  x_axis="time",
                                  y_axis="linear",
                                  ax=ax[2],
                                  cmap="twilight")
  ax[2].set_title("Phase Correction Angle (radians)")
  fig.colorbar(img2, ax=ax[2], format="%+0.2f rad")

  plt.tight_layout()
  plt.show()

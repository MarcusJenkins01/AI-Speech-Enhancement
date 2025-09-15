import random
import shutil
import glob
import os
import tqdm


def split_GRID(dataset_dir, out_dir, train_ratio=0.7, seed=0):
  os.makedirs(out_dir, exist_ok=True)

  speaker_ids = glob.glob(os.path.join(dataset_dir, "*"))
  rng = random.Random(seed)
  rng.shuffle(speaker_ids)

  train_count = int(len(speaker_ids) * train_ratio)
  val_count = (len(speaker_ids) - train_count) // 2
  val_end = train_count + val_count
  split_ids = {
    "train": speaker_ids[:train_count],
    "val": speaker_ids[train_count:val_end],
    "test": speaker_ids[val_end:]
  }

  for split, id_paths in split_ids.items():
    for id_path in tqdm.tqdm(id_paths, desc=f"Generating {split} split"):
      id_dest = os.path.join(out_dir, split, os.path.basename(id_path))
      shutil.copytree(id_path, id_dest, dirs_exist_ok=True)


def split_DEMAND(dataset_dir, out_dir, train_ratio=0.7, seed=0, audio_ext=".wav"):
  audio_paths = glob.glob(os.path.join(dataset_dir, f"*/*{audio_ext}"))
  rng = random.Random(seed)
  rng.shuffle(audio_paths)

  train_count = int(len(audio_paths) * train_ratio)
  val_count = (len(audio_paths) - train_count) // 2
  val_end = train_count + val_count
  split_paths = {
    "train": audio_paths[:train_count],
    "val": audio_paths[train_count:val_end],
    "test": audio_paths[val_end:]
  }

  for split, a_paths in split_paths.items():
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    for a_path in tqdm.tqdm(a_paths, desc=f"Generating {split} split"):
      s_dir = os.path.basename(os.path.dirname(a_path)).lower()
      a_dest = os.path.join(split_dir, f"{s_dir}_{os.path.basename(a_path)}")
      shutil.copy(a_path, a_dest)


if __name__ == "__main__":
  split_GRID("D:/Datasets/GRID/Full", "D:/Datasets/GRID/Splits")
  split_DEMAND("D:/Datasets/DEMAND/Full", "D:/Datasets/DEMAND/Splits")

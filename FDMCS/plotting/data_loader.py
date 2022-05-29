import os
import numpy as np

class DataLoader():
  def __init__(self, dir):
    self.dir = dir

    self.files = os.listdir(self.dir)
    self.files.sort(key=self.extract_time)

  def __getitem__(self, idx):
    particles = []
    duration = None
    with open(os.path.join(self.dir, self.files[idx])) as f:
      duration = float(f.readline().strip())
      for line in f:
          particles.append(list(map(float, line.strip().split())))

    sizes, counts, _ = zip(*particles)
    sizes = np.array(sizes)
    counts = np.array(counts)
    return sizes, counts / np.sum(counts * sizes), self.extract_time(self.files[idx]), duration

  def __len__(self):
    return len(self.files)

  @staticmethod
  def extract_time(filename):
    return float(filename.replace('.cpt', ''))


class FDMCSDataLoader():
  def __init__(self, dir):
    self.dir = dir

    self.files = os.listdir(self.dir)
    self.files.sort(key=self.extract_time)

  def __getitem__(self, idx):
    particles = []
    duration = None
    with open(os.path.join(self.dir, self.files[idx])) as f:
      duration = float(f.readline().strip())
      for line in f:
          particles.append(list(map(float, line.strip().split())))

    sizes, counts, _ = zip(*particles)
    sizes, counts = self.aggregate_counts(sizes, counts)
    return sizes, counts / np.sum(counts * sizes), self.extract_time(self.files[idx]), duration

  def __len__(self):
    return len(self.files)

  @staticmethod
  def extract_time(filename):
    return float(filename.replace('.cpt', ''))

  @staticmethod
  def aggregate_counts(sizes, counts):
    aggr = defaultdict(int)
    for i in range(len(sizes)):
        aggr[sizes[i]] += counts[i]
    return np.array(list(aggr.keys())), np.array(list(aggr.values()))

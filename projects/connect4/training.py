from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from experiments import Transition, Experiment
from games import RandomStrat
from typing import List


class TransitionDataset(Dataset):
    def __init__(self, samples: List[Transition]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return {
            "boards": torch.tensor(s.boards, dtype=torch.float32),
            "action": torch.tensor(s.action, dtype=torch.long),
            "dc_return": torch.tensor(s.dc_return, dtype=torch.float32),
        }

    def __getitems__(self, idxs: List):
        # improve to run faster
        samples = []
        for idx in idxs:
            samples.append(self.__getitem__(idx))
        return samples


class Connect4Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x


## training
# exp01 = Experiment(RandomStrat(42))
# exp01.play(10000)
# exp01.build_features()
# dataset = TransitionDataset(exp01.samples)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# exp01.samples[-1].boards[0]
# exp01.samples[-1].boards[1]

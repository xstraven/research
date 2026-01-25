from torch.utils.data import Dataset, DataLoader
import torch
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


## training
exp01 = Experiment(RandomStrat(42))
exp01.play(10000)
exp01.build_features()
dataset = TransitionDataset(exp01.samples)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

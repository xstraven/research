from __future__ import annotations
from dataclasses import dataclass
from experiments import Experiment
from typing import List


@dataclass
class Transition:
    board: List[List]
    player: int
    action: int
    dc_return: int


class DataLoader:
    def __init__(self, experiment: Experiment, gamma=0.99, batch_size=32):
        self.gamma = gamma
        self.batch_size = batch_size
        self.experiment = experiment

    def build_features(self):
        pass

    def yield_batches(self):
        pass

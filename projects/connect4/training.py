from __future__ import annotations
from dataclasses import dataclass
from experiments import Experiment
from typing import List
from games import get_empty_board
from copy import deepcopy


@dataclass
class Transition:
    boards: List
    player: int
    action: int
    dc_return: int


class DataLoader:
    def __init__(self, experiment: Experiment, gamma=0.95, reward=1):
        self.gamma: float = gamma
        self.reward: int = reward
        self.experiment: Experiment = experiment
        self.samples: List[Transition] = []

    def build_features(self):
        for game in self.experiment.games:
            boards = [get_empty_board(), get_empty_board()]
            dc_return = self.reward * (self.gamma ** (len(game.moves) - 1))
            for move in game.moves:
                player = move[0]
                x, y = move[1], move[2]

                t = Transition(
                    boards=deepcopy(boards),
                    player=player,
                    action=y,
                    dc_return=dc_return,
                )
                self.samples.append(t)
                boards[0 if player == 1 else 1][x][y] = 1 if player == 1 else -1
                dc_return = (1 if player == 1 else -1) * (
                    dc_return / self.gamma
                )

                # t = Transition(board=, p, a, dc_return)

    def yield_batches(self):
        pass

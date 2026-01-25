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
            player1 = game.moves[0][0]
            player2 = game.moves[1][0]
            dc_return = self.reward * (self.gamma ** (len(game.moves) - 1))
            if game.winner == 0:
                dc_return = 0
            if game.winner == player2:
                dc_return = -dc_return
            for move in game.moves:
                player = move[0]
                x, y = move[1], move[2]
                player_return = dc_return if player == player1 else -dc_return
                player_boards = (
                    deepcopy(boards)
                    if player == player1
                    else deepcopy(boards[::-1])
                )
                player_boards[1] = [
                    [-cell for cell in row] for row in player_boards[1]
                ]
                t = Transition(
                    boards=player_boards,
                    player=player,
                    action=y,
                    dc_return=player_return,
                )
                self.samples.append(t)
                boards[0 if player == player1 else 1][x][y] = 1
                dc_return = dc_return / self.gamma

                # t = Transition(board=, p, a, dc_return)

    def yield_batches(self):
        pass

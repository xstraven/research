from __future__ import annotations
from typing import List
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Game:
    states: List
    moves: List
    legal_moves: dict
    winner: int

    @classmethod
    def empty(cls) -> Game:
        board = get_empty_game()
        moves = []
        winner = 0
        legal_moves = {col: 5 for col in range(7)}
        return cls(
            states=board, moves=moves, winner=winner, legal_moves=legal_moves
        )

    def apply_move(self, y, val) -> None:
        x = self.legal_moves[y]
        self.states[x][y] = val
        self.moves.append([val, x, y])
        self.legal_moves[y] -= 1
        if self.legal_moves[y] <= 0:
            del self.legal_moves[y]

    def has_winner(self) -> bool:
        board_state = self.states
        _, x, y = self.moves[-1]
        val = board_state[x][y]

        # check for vertical win
        if x < 3:
            if (
                val == board_state[x + 1][y]
                and val == board_state[x + 2][y]
                and val == board_state[x + 3][y]
            ):
                self.winner = val
                return True

        # check for horizontal win
        for i in range(6):
            for j in range(0, 4):
                if board_state[i][j : j + 4] == [val, val, val, val]:
                    self.winner = val
                    return True

        # check for diagonal win (bottom-left to top-right)
        for i in range(3, 6):
            for j in range(0, 4):
                if (
                    board_state[i][j] == val
                    and board_state[i - 1][j + 1] == val
                    and board_state[i - 2][j + 2] == val
                    and board_state[i - 3][j + 3] == val
                ):
                    self.winner = val
                    return True

        # check for diagonal win (top-left to bottom-right)
        for i in range(0, 3):
            for j in range(0, 4):
                if (
                    board_state[i][j] == val
                    and board_state[i + 1][j + 1] == val
                    and board_state[i + 2][j + 2] == val
                    and board_state[i + 3][j + 3] == val
                ):
                    self.winner = val
                    return True

        return False


def get_empty_game() -> List[List]:
    return [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]


class Policy(ABC):
    def play_game(self, game: Game) -> Game:
        player = 1
        while game.legal_moves:
            move = self.choose_move(game, player)
            game.apply_move(move, player)
            if game.has_winner():
                break
            player = 2 if player == 1 else 1
        return game

    @abstractmethod
    def choose_move(self, game, player: int) -> Game: ...

    @abstractmethod
    def config(self) -> dict: ...


class RandomStrat(Policy):
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)

    def choose_move(self, game: Game, player: int) -> Game:
        return self.rng.choice(list(game.legal_moves.keys()))

    def config(self) -> dict:
        return {"seed": self.seed}

from __future__ import annotations
from games import RandomStrat, Game, Policy, get_empty_board
from copy import deepcopy
from typing import List
from datetime import datetime as dt
from utils import (
    DATA_FOLDER,
    load_games_parquet,
    load_metadata_json,
    save_games_parquet,
    save_metadata_json,
)
from dataclasses import dataclass, field
import time


@dataclass
class Transition:
    boards: List
    player: int
    action: int
    dc_return: int


@dataclass
class Experiment:
    policy: Policy
    games: List[Game] = field(default_factory=list, init=False)
    samples: List[Transition] = field(default_factory=list, init=False)

    def play(self, n_games) -> None:
        t1 = time.time()
        strat = self.policy
        for i in range(n_games):
            game = Game()
            strat.play_game(game)
            self.games.append(game)

        print(f"Time: {time.time() - t1:.2f}s for {n_games} games.")

    def save(self, name: str) -> None:
        date = dt.now().strftime(format="%Y%m%d_%H:%M")
        filename = name + "_" + date
        base_path = DATA_FOLDER / filename
        save_games_parquet(base_path.with_suffix(".parquet"), self.games)
        save_metadata_json(
            base_path.with_suffix(".json"),
            {
                "policy": type(self.policy).__name__,
                "policy_config": self.policy.config(),
            },
        )
        print(f"saved as {base_path} .json and .parquet")

    @classmethod
    def load(cls, name: str, policy: Policy) -> Experiment:
        games = load_games_parquet(name)
        metadata = load_metadata_json(name)
        instance = cls(
            metadata.get("n_games", len(games)), policy, metadata.get("rseed")
        )
        instance.games = games
        return instance

    @property
    def win_rate(self) -> List[float]:
        if not self.games:
            print("No games played")
            return [0, 0, 0]

        total_games = len(self.games)
        p1_wins = 0
        p2_wins = 0
        for game in self.games:
            if game.winner == 1:
                p1_wins += 1
            elif game.winner == 2:
                p2_wins += 1

        p1_pwin = p1_wins / total_games
        p2_pwin = p2_wins / total_games

        print(f"Player 1 won {p1_pwin:.2%}% of games.")
        print(f"Player 2 won {p2_pwin:.2%}% of games.")
        print(f"{1-p1_pwin - p2_pwin:.2%} games ended in a draw.")
        return [p1_pwin, p2_pwin, 1 - p1_pwin - p2_pwin]

    def build_features(self, reward=1, gamma=0.95):
        for game in self.games:
            boards = [get_empty_board(), get_empty_board()]
            player1 = game.moves[0][0]
            player2 = game.moves[1][0]
            dc_return = reward * (gamma ** (len(game.moves) - 1))
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
                    [min(1, cell) for cell in row] for row in player_boards[1]
                ]
                t = Transition(
                    boards=player_boards,
                    player=player,
                    action=y,
                    dc_return=player_return,
                )
                self.samples.append(t)
                boards[0 if player == player1 else 1][x][y] = 1
                dc_return = dc_return / gamma


def main():
    exp01 = Experiment(RandomStrat(42))
    exp01.play(10000)
    _ = exp01.win_rate
    exp01.save("first_random")

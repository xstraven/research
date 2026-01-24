from __future__ import annotations
from games import RandomStrat, Game, Policy
from typing import List
from datetime import datetime as dt
from data import (
    DATA_FOLDER,
    load_games_parquet,
    load_metadata_json,
    save_games_parquet,
    save_metadata_json,
)
from dataclasses import dataclass, field
import time


@dataclass
class Experiment:
    policy: Policy
    games: list[Game] = field(default_factory=list, init=False)

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


def main():
    exp01 = Experiment(RandomStrat(42))
    exp01.play(100000)
    _ = exp01.win_rate
    exp01.save("first_random")

    # games_save = (
    #     f"random_100k_{dt.datetime.now().strftime(format="%Y-%m-%d_%H:%M")}"
    # )
    # games.save(name=games_save)
    # first_random_20260124_21:14


if __name__ == "__main__":
    main()

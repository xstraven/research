from __future__ import annotations

from pathlib import Path
import json
import polars as pl
import numpy as np
from dataclasses import dataclass
from sim_games import Game

# Point to the data folder at the root of the repository
DATA_FOLDER = Path(__file__).parent.parent.parent / "data" / "connect4"
MODELS_FOLDER = Path(__file__).parent.parent.parent / "models" / "connect4"


@dataclass
class Transition:
    state: np.array
    player: int
    action: int
    disc_return: int


def _ensure_data_folder() -> None:
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)


def save_games_parquet(path: Path, games: list[Game]) -> None:
    _ensure_data_folder()
    df = pl.DataFrame(
        {
            "winner": [game.winner for game in games],
            "moves": [game.moves for game in games],
        }
    )
    df.write_parquet(path)


def load_games_parquet(path: Path) -> list[Game]:
    df = pl.read_parquet(path)
    games: list[Game] = []
    for row in df.iter_rows(named=True):
        game = Game.empty()
        for move in row["moves"]:
            game.apply_move(move[2], move[0])
        game.winner = row["winner"]
        games.append(game)
    return games


def save_metadata_json(path: Path, metadata: dict) -> None:
    _ensure_data_folder()
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def load_metadata_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class DataLoader:
    def __init__(self, gamma=0.99, batch_size=32, shuffle=True):
        self.gamma = gamma
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data: list[Transition] = []

    def build_features(self):
        pass

    def yield_batches(self):
        pass

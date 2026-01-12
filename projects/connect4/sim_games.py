import polars as pl
from data import DATA_FOLDER
from typing import List
from pydantic import BaseModel
import random


class Game(BaseModel):
    pass


def get_empty_game() -> List[List]:
    return [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]


def check_victory(board_state: List[List], last_played: tuple) -> bool:
    x, y = last_played
    val = board_state[x][y]

    # check for vertical win
    if x < 3:
        if (
            val == board_state[x + 1][y]
            and val == board_state[x + 2][y]
            and val == board_state[x + 3][y]
        ):
            return True

    # check for horizontal win
    for i in range(6):
        for j in range(0, 4):
            if board_state[i][j : j + 4] == [val, val, val, val]:
                return True

    # check for diagonal wins (bottom-left to top-right)
    for i in range(3, 6):
        for j in range(0, 4):
            if (
                board_state[i][j] == val
                and board_state[i - 1][j + 1] == val
                and board_state[i - 2][j + 2] == val
                and board_state[i - 3][j + 3] == val
            ):
                return True

    # check for diagonal wins (top-left to bottom-right)
    for i in range(0, 3):
        for j in range(0, 4):
            if (
                board_state[i][j] == val
                and board_state[i + 1][j + 1] == val
                and board_state[i + 2][j + 2] == val
                and board_state[i + 3][j + 3] == val
            ):
                return True

    return False


def sim_game():
    state = get_empty_game()
    val = 1
    moves_played = []
    moves_legal = {col: 5 for col in range(7)}

    while len(moves_legal) > 0:
        y = random.choice(list(moves_legal.keys()))
        x = moves_legal[y]
        state[x][y] = val
        moves_played.append([val, x, y])
        if check_victory(state, (x, y)):
            return state, moves_played, True
        val = 2 if val == 1 else 1
        moves_legal[y] -= 1
        if moves_legal[y] <= 0:
            del moves_legal[y]

    return state, moves_played, False


def main():
    import time

    t1 = time.time()
    games = 0
    while games < 10000:
        has_winner = False
        no_winners = -1
        while not has_winner:
            no_winners += 1
            game_end, moves, has_winner = sim_game()
        games += 1
    print(time.time() - t1)
    # print(f"Winner winner! after {no_winners} games.")
    # for row in game_end:
    #     print(row)
    # print(moves)


if __name__ == "__main__":
    main()

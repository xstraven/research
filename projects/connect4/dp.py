from games import Policy, Game


class DPStrat(Policy):
    def __init__(self):
        self._cache = {}

    def choose_move(self, game: Game, player: int) -> Game:
        return self.rng.choice(list(game.legal_moves.keys()))

    def config(self) -> dict:
        return {}


## solving
l1 = [[1, 2, 3], [4, 5, 6]]
l2 = [[2, 3, 4], [5, 6, 7]]
s1 = set([l1, l2])

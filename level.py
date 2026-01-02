from dataclasses import dataclass
from typing import FrozenSet

Pos = tuple[int, int]  # (x, y)

DIRS: dict[str, Pos] = {
    "U": (0, -1),
    "D": (0,  1),
    "L": (-1, 0),
    "R": (1,  0),
}

def add_pos(a: Pos, b: Pos) -> Pos:
    return (a[0] + b[0], a[1] + b[1])

@dataclass(frozen=True)
class State:
    player: Pos
    boxes: FrozenSet[Pos]

@dataclass(frozen=True)
class Level:
    width: int
    height: int
    walls: frozenset[Pos]
    goals: frozenset[Pos]

    @classmethod
    def from_lines(cls, lines: list[str]) -> tuple["Level", State]:
        h = len(lines)
        w = max(len(r) for r in lines) if h else 0
        norm = [r.ljust(w) for r in lines]

        walls: set[Pos] = set()
        goals: set[Pos] = set()
        boxes: set[Pos] = set()
        player: Pos | None = None

        cells = [((x, y), ch)
                for y, row in enumerate(norm)
                for x, ch in enumerate(row)]

        walls = {p for p, ch in cells if ch == 'X'}
        goals = {p for p, ch in cells if ch in {'.', '&', '+'}}
        boxes = {p for p, ch in cells if ch in {'*', '&'}}
        players = [p for p, ch in cells if ch in {'@', '+'}]
        
        if len(players) != 1:
            raise ValueError("Mapa musi mieć dokładnie 1 gracza (@ lub +)")
        player = players[0]


        if player is None:
            raise ValueError("Brak gracza '@' (lub '+') na mapie")
        if not boxes:
            raise ValueError("Brak skrzynek '$' (lub '*') na mapie")
        if not goals:
            raise ValueError("Brak celów '.' (lub '*' / '+') na mapie")

        level = cls(w, h, frozenset(walls), frozenset(goals))
        start = State(player, frozenset(boxes))
        return level, start

    def is_solved(self, state: State) -> bool:
        return state.boxes.issubset(self.goals)

    def step(self, state: State, action: str) -> tuple[State, bool] | None:
        if action not in DIRS:
            return None

        d = DIRS[action]
        np = add_pos(state.player, d)

        if np in self.walls:
            return None

        boxes = set(state.boxes)

        if np in boxes:  # pchanie
            nb = add_pos(np, d)
            if nb in self.walls or nb in boxes:
                return None
            boxes.remove(np)
            boxes.add(nb)
            return State(np, frozenset(boxes)), True

        return State(np, state.boxes), False

    def legal_moves(self, state: State) -> list[str]:
        out = []
        for a in DIRS:
            if self.step(state, a) is not None:
                out.append(a)
        return out



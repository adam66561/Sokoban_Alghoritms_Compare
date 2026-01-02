from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import time
from typing import Optional, Callable
from level import Level, State


@dataclass(frozen=True)
class SolveResult:
    solved: bool
    moves: list[str]          
    visited_states: int
    expanded_states: int
    time_ms: float
    pushes: int

    last_state: Optional[State]  
    last_moves: list[str]        


def reconstruct_moves(
    goal: State,
    parent: dict[State, State],
    parent_action: dict[State, str],
    start: State,
) -> list[str]:
    moves: list[str] = []
    cur = goal
    while cur != start:
        moves.append(parent_action[cur])
        cur = parent[cur]
    moves.reverse()
    return moves


def count_pushes(level: Level, start: State, moves: list[str]) -> int:
    pushes = 0
    s = start
    for a in moves:
        res = level.step(s, a)
        if res is None:
            break
        s, pushed = res
        pushes += int(pushed)
    return pushes


def bfs_solve(
    level: Level,
    start: State,
    max_states: Optional[int] = None,
    on_progress: Optional[Callable[[State, int, int, int, float], None]] = None,
) -> SolveResult:
    """
    BFS po stanach: (pozycja gracza, pozycje skrzynek).
    Zwraca najkrótsze rozwiązanie w liczbie ruchów 

    max_states: limit odwiedzonych stanów 
    max_time_s: limit czasu w sekundach
    """
    t0 = time.perf_counter()
    depth: dict[State, int] = {start: 0}

    last_state: State = start
    last_moves: list[str] = []

    q = deque([start])
    visited: set[State] = {start}

    parent: dict[State, State] = {}
    parent_action: dict[State, str] = {}

    expanded = 0

    layer_left = 1          
    next_layer_count = 0  

    if level.is_solved(start):
        return SolveResult(True, [], 1, 0, 0.0, 0)

    while q:
        if max_states is not None and len(visited) >= max_states:
            break

        s = q.popleft()
        expanded += 1

        d = depth[s]

        last_state = s
        last_moves = reconstruct_moves(s, parent, parent_action, start) if s != start else []

        for a in level.legal_moves(s):
            res = level.step(s, a)
            if res is None:
                continue
            ns, _pushed = res

            if ns in visited:
                continue

            visited.add(ns)
            parent[ns] = s
            parent_action[ns] = a
            depth[ns] = d + 1

            next_layer_count += 1

            if level.is_solved(ns):
                moves = reconstruct_moves(ns, parent, parent_action, start)
                pushes = count_pushes(level, start, moves)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                return SolveResult(True, moves, len(visited), expanded, dt_ms, pushes, ns, moves)

            q.append(ns)
        
        layer_left -= 1
        layer_end = (layer_left == 0)
        if layer_end:
            layer_left = next_layer_count
            next_layer_count = 0

        if on_progress:
            dt = time.perf_counter() - t0
            on_progress(s, d, len(visited), expanded, dt, layer_end)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return SolveResult(False, [], len(visited), expanded, dt_ms, 0, last_state, last_moves)



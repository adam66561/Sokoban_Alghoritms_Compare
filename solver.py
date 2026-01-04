from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import time
from typing import Optional, Callable
from level import Level, State
import heapq
import math

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


def is_corner_deadlock(level: Level, s: State) -> bool:
    walls = level.walls
    goals = level.goals
    for (bx, by) in s.boxes:
        if (bx, by) in goals:
            continue
        # dwa prostopadłe "mury" -> róg
        if ((bx-1, by) in walls and (bx, by-1) in walls): return True
        if ((bx-1, by) in walls and (bx, by+1) in walls): return True
        if ((bx+1, by) in walls and (bx, by-1) in walls): return True
        if ((bx+1, by) in walls and (bx, by+1) in walls): return True
    return False


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

    last_moves = reconstruct_moves(s, parent, parent_action, start) if s != start else []
    return SolveResult(False, [], len(visited), expanded, dt_ms, 0, last_state, last_moves)


def dfs_solve(
    level: Level,
    start: State,
    max_depth: int = 200,
    max_states: int | None = None,
    on_progress: Optional[Callable[[State, int, int, int, float], None]] = None,
) -> SolveResult:
    t0 = time.perf_counter()

    stack = [(start, 0)]
    visited: set[State] = {start}

    parent: dict[State, State] = {}
    parent_action: dict[State, str] = {}

    expanded = 0
    last_state = start

    if level.is_solved(start):
        return SolveResult(True, [], 1, 0, 0.0, 0, start, [])

    while stack:
        s, depth = stack.pop()
        expanded += 1
        last_state = s

        if max_states is not None and len(visited) >= max_states:
            break

        if on_progress:
            dt = time.perf_counter() - t0
            on_progress(s, depth, len(visited), expanded, dt, False)

        if depth >= max_depth:
            continue

        for a in level.legal_moves(s):
            res = level.step(s, a)
            if res is None:
                continue

            ns, _ = res
            if ns in visited:
                continue

            visited.add(ns)
            parent[ns] = s
            parent_action[ns] = a

            if level.is_solved(ns):
                moves = reconstruct_moves(ns, parent, parent_action, start)
                pushes = count_pushes(level, start, moves)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                return SolveResult(True, moves, len(visited), expanded, dt_ms, pushes, ns, moves)

            stack.append((ns, depth + 1))

    dt_ms = (time.perf_counter() - t0) * 1000.0
    last_moves = reconstruct_moves(last_state, parent, parent_action, start) if last_state != start else []
    return SolveResult(False, [], len(visited), expanded, dt_ms, 0, last_state, last_moves)



def heuristic_manhattan_to_goals(level: Level, s: State) -> int:
    # admissible lower bound: suma odległości każdej skrzynki do najbliższego celu (ignoruje kolizje między skrzynkami)
    goals = list(level.goals)
    if not goals:
        return 0
    h = 0
    for (bx, by) in s.boxes:
        best = min(abs(bx - gx) + abs(by - gy) for (gx, gy) in goals)
        h += best
    return h

def a_star_solve(
    level: Level,
    start: State,
    max_states: Optional[int] = None,
    on_progress: Optional[Callable[[State, int, int, int, float, bool], None]] = None,
) -> SolveResult:
    """
    A* po stanach (player, boxes). Koszt = liczba ruchów.
    Heurystyka: suma Manhattan(box -> najbliższy goal) (dolne ograniczenie).
    """
    t0 = time.perf_counter()

    if level.is_solved(start):
        return SolveResult(True, [], 1, 0, 0.0, 0, start, [])

    parent: dict[State, State] = {}
    parent_action: dict[State, str] = {}

    g_score: dict[State, int] = {start: 0}
    closed: set[State] = set()

    h0 = heuristic_manhattan_to_goals(level, start)
    counter = 0
    heap: list[tuple[int, int, int, State]] = [(h0, 0, counter, start)]  # (f, g, tie, state)

    expanded = 0
    last_state: State = start

    while heap:
        if max_states is not None and (len(g_score) >= max_states):
            break

        f, g, _, s = heapq.heappop(heap)
        if s in closed:
            continue

        closed.add(s)
        expanded += 1
        last_state = s

        if on_progress:
            dt = time.perf_counter() - t0
            # "depth" dla GUI = g (liczba ruchów)
            on_progress(s, g, len(g_score), expanded, dt, False)

        if level.is_solved(s):
            moves = reconstruct_moves(s, parent, parent_action, start)
            pushes = count_pushes(level, start, moves)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return SolveResult(True, moves, len(g_score), expanded, dt_ms, pushes, s, moves)

        # rozwijaj następniki
        for a in level.legal_moves(s):
            res = level.step(s, a)
            if res is None:
                continue
            ns, pushed = res

            # proste odcinanie deadlocków: sprawdzaj tylko gdy było pchnięcie
            if pushed and is_corner_deadlock(level, ns):
                continue

            ng = g + 1
            old = g_score.get(ns)
            if old is not None and ng >= old:
                continue

            g_score[ns] = ng
            parent[ns] = s
            parent_action[ns] = a

            counter += 1
            nh = heuristic_manhattan_to_goals(level, ns)
            nf = ng + nh
            heapq.heappush(heap, (nf, ng, counter, ns))

    dt_ms = (time.perf_counter() - t0) * 1000.0
    last_moves = reconstruct_moves(last_state, parent, parent_action, start) if last_state != start else []
    return SolveResult(False, [], len(g_score), expanded, dt_ms, 0, last_state, last_moves)

    
def gbfs_solve(
    level: Level,
    start: State,
    max_states: Optional[int] = None,
    on_progress: Optional[Callable[[State, int, int, int, float, bool], None]] = None,
) -> SolveResult:
    """
    Greedy Best-First Search po stanach (player, boxes).
    Wybiera zawsze stan z najmniejszą heurystyką h (ignoruje g).
    Nie gwarantuje najkrótszego rozwiązania.
    """
    t0 = time.perf_counter()

    if level.is_solved(start):
        return SolveResult(True, [], 1, 0, 0.0, 0, start, [])

    parent: dict[State, State] = {}
    parent_action: dict[State, str] = {}

    # GBFS: zwykłe visited/closed wystarczy
    closed: set[State] = set([start])

    # g_depth trzymamy tylko do UI i ewentualnego last_moves
    depth: dict[State, int] = {start: 0}

    h0 = heuristic_manhattan_to_goals(level, start)
    counter = 0

    # GBFS: w heapie priorytetem jest samo h
    heap: list[tuple[int, int, State]] = [(h0, counter, start)]  # (h, tie, state)

    expanded = 0
    last_state: State = start

    while heap:
        if max_states is not None and len(closed) >= max_states:
            break

        h, _, s = heapq.heappop(heap)
        expanded += 1
        last_state = s

        g = depth[s]  # tylko informacyjnie

        if on_progress:
            dt = time.perf_counter() - t0
            on_progress(s, g, len(closed), expanded, dt, False)

        if level.is_solved(s):
            moves = reconstruct_moves(s, parent, parent_action, start)
            pushes = count_pushes(level, start, moves)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return SolveResult(True, moves, len(closed), expanded, dt_ms, pushes, s, moves)

        for a in level.legal_moves(s):
            res = level.step(s, a)
            if res is None:
                continue
            ns, pushed = res

            if pushed and is_corner_deadlock(level, ns):
                continue

            if ns in closed:
                continue

            closed.add(ns)
            parent[ns] = s
            parent_action[ns] = a
            depth[ns] = g + 1

            counter += 1
            nh = heuristic_manhattan_to_goals(level, ns)
            heapq.heappush(heap, (nh, counter, ns))

    dt_ms = (time.perf_counter() - t0) * 1000.0
    last_moves = reconstruct_moves(last_state, parent, parent_action, start) if last_state != start else []
    return SolveResult(False, [], len(closed), expanded, dt_ms, 0, last_state, last_moves)

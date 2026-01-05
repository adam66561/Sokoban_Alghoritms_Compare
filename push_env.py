from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque
import numpy as np # type: ignore

from level import Level, State
from solver import is_corner_deadlock, heuristic_manhattan_to_goals

DIRS = {
    "U": (0, -1),
    "D": (0,  1),
    "L": (-1, 0),
    "R": (1,  0),
}
DIR_LIST = ["U", "D", "L", "R"]

@dataclass
class StepOut:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict

class PushSokobanEnv:
    """
    Push-based env:
    - akcja = (cell_index, dir) zakodowana jako idx = (y*w + x)*4 + dir_idx
    - legalność akcji sprawdzana maską (action_mask)
    - wykonanie akcji: agent "teleportuje się logicznie" poprzez walk + push
      (środowisko generuje sekwencję UDLR w info['moves'] żebyś mógł to animować w GUI)
    """

    def __init__(self, level: Level, start: State, max_pushes: int = 80):
        self.level = level
        self.start = start
        self.max_pushes = max_pushes

        self.visited_states = set()

        self.w, self.h = level.width, level.height
        self.n_actions = self.w * self.h * 4

        # stałe kanały
        self._walls = self._grid(level.walls)
        self._goals = self._grid(level.goals)

        self.reset()

    def _grid(self, pos_set) -> np.ndarray:
        g = np.zeros((self.h, self.w), dtype=np.float32)
        for (x, y) in pos_set:
            g[y, x] = 1.0
        return g

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.state = self.start
        self.pushes = 0

        self.visited_states = {(self.state.player, tuple(sorted(self.state.boxes)))}

        self.prev_h = heuristic_manhattan_to_goals(self.level, self.state)
        self.prev_on_goals = self._boxes_on_goals()

        obs = self._obs()
        mask, push_map = self._action_mask_and_map(self.state)
        self._last_push_map = push_map  # idx -> (walk_moves, push_move)
        return obs, mask

    def _boxes_on_goals(self) -> int:
        return sum(1 for b in self.state.boxes if b in self.level.goals)

    def _obs(self) -> np.ndarray:
        boxes = np.zeros((self.h, self.w), dtype=np.float32)
        for (x, y) in self.state.boxes:
            boxes[y, x] = 1.0

        player = np.zeros((self.h, self.w), dtype=np.float32)
        px, py = self.state.player
        player[py, px] = 1.0

        # (C,H,W)
        return np.stack([self._walls, self._goals, boxes, player], axis=0)

    # --------- WALK BFS (bez pchania) ----------
    def _walk_bfs_parents(self, s: State) -> Dict[Tuple[int,int], Tuple[Tuple[int,int], str]]:
        """BFS po polach osiągalnych przez gracza bez pchania skrzynek.
        Zwraca parenty do rekonstrukcji ścieżki walk."""
        start = s.player
        boxes = set(s.boxes)
        walls = self.level.walls

        q = deque([start])
        parents: Dict[Tuple[int,int], Tuple[Tuple[int,int], str]] = {start: (start, "")}

        while q:
            x, y = q.popleft()
            for a, (dx, dy) in DIRS.items():
                nx, ny = x + dx, y + dy
                np_ = (nx, ny)
                if np_ in parents:
                    continue
                if np_ in walls or np_ in boxes:
                    continue
                parents[np_] = ((x, y), a)
                q.append(np_)
        return parents

    def _reconstruct_walk(self, parents, target: Tuple[int,int]) -> List[str] | None:
        if target not in parents:
            return None
        cur = target
        moves: List[str] = []
        while True:
            prev, a = parents[cur]
            if cur == prev:
                break
            moves.append(a)
            cur = prev
        moves.reverse()
        return moves

    # --------- akcje PUSH ----------
    def _action_index(self, x: int, y: int, dir_idx: int) -> int:
        return (y * self.w + x) * 4 + dir_idx

    def _decode_action(self, idx: int) -> Tuple[int,int,int]:
        cell = idx // 4
        dir_idx = idx % 4
        x = cell % self.w
        y = cell // self.w
        return x, y, dir_idx

    def _action_mask_and_map(self, s: State) -> Tuple[np.ndarray, Dict[int, Tuple[List[str], str]]]:
        """
        mask[idx]=1 jeśli możliwe jest PUSH skrzynki stojącej na (x,y) w dir
        push_map[idx] = (walk_moves_to_behind, push_move)
        """
        mask = np.zeros((self.n_actions,), dtype=np.float32)
        push_map: Dict[int, Tuple[List[str], str]] = {}

        boxes = set(s.boxes)
        walls = self.level.walls

        parents = self._walk_bfs_parents(s)

        for (bx, by) in boxes:
            for dir_idx, push_move in enumerate(DIR_LIST):
                dx, dy = DIRS[push_move]

                # gdzie gracz musi stanąć, aby popchnąć: za skrzynką
                behind = (bx - dx, by - dy)
                # gdzie skrzynka poleci: przed skrzynką
                front = (bx + dx, by + dy)

                # front musi być wolny (nie ściana, nie skrzynka)
                if front in walls or front in boxes:
                    continue
                # behind musi być osiągalne pieszo
                walk_moves = self._reconstruct_walk(parents, behind)
                if walk_moves is None:
                    continue

                # dodatkowo: behind nie może być ścianą/skrzynką 
                if behind in walls or behind in boxes:
                    continue

                idx = self._action_index(bx, by, dir_idx)
                mask[idx] = 1.0
                push_map[idx] = (walk_moves, push_move)

        return mask, push_map

    def step(self, action_idx: int) -> Tuple[StepOut, np.ndarray]:
        """
        Zwraca:
          StepOut + next_action_mask
        """
        self.pushes += 1
        done = False
        reward = -2.0  # koszt pchnięcia (macro-step)
        info: Dict = {}

        state_hash = (self.state.player, tuple(sorted(self.state.boxes)))
        if state_hash in self.visited_states:
            reward -= 2.0  # kara za kręcenie się w kółko
        self.visited_states.add(state_hash)

        # kara i koniec gdy akcja nielegalna
        if action_idx not in self._last_push_map:
            reward -= 10.0
            obs = self._obs()
            next_mask, next_map = self._action_mask_and_map(self.state)
            self._last_push_map = next_map
            return StepOut(obs, reward, False, {"illegal": True}), next_mask

        walk_moves, push_move = self._last_push_map[action_idx]

        # wykonaj WALK
        s = self.state
        for m in walk_moves:
            res = self.level.step(s, m)
            if res is None:
                reward -= 10.0
                obs = self._obs()
                next_mask, next_map = self._action_mask_and_map(self.state)
                self._last_push_map = next_map
                return StepOut(obs, reward, True, {"walk_failed": True}), next_mask
            s, pushed = res
            if pushed:
                # walk nie powinien pchać
                reward -= 10.0

        # wykonaj PUSH
        res = self.level.step(s, push_move)
        if res is None:
            reward -= 10.0
            self.state = s
        else:
            ns, pushed = res
            self.state = ns

            # shaping: skrzynki na celach
            old_on = self.prev_on_goals
            new_on = self._boxes_on_goals()
            if new_on > old_on:
                reward += 150.0
            elif new_on < old_on:
                reward -= 150.0
            self.prev_on_goals = new_on

            # deadlock po pchnięciu
            if pushed and is_corner_deadlock(self.level, self.state):
                reward -= 10.0
                done = True

            # heurystyka delta 
            h = heuristic_manhattan_to_goals(self.level, self.state)
            reward += 5.0 * (self.prev_h - h)
            self.prev_h = h

            if self.level.is_solved(self.state):
                reward += 1000.0
                done = True

        if self.pushes >= self.max_pushes:
            done = True

        # przygotuj next maskę
        obs = self._obs()
        next_mask, next_map = self._action_mask_and_map(self.state)
        self._last_push_map = next_map

        # sekwencja UDLR do animacji
        info["moves"] = walk_moves + [push_move]
        info["pushes"] = self.pushes

        return StepOut(obs, reward, done, info), next_mask
